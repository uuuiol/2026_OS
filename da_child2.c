/*
 * ============================================================
 *  da_child2.c  —  다 실험: PARENT1 + CHILD2 병렬
 *
 *  세팅 단계:
 *    PARENT : 조도 계산 + 몬스터AI + 파티클 시뮬
 *    CHILD1 : 청크 0~127  (256청크 중 절반) noise→merge→blur→erosion→histogram
 *    CHILD2 : 청크 128~255 (256청크 중 절반)
 *    → 두 Child 완료 후 Parent가 두 지형 조각을 병합 → 통합 blur/erosion/histogram
 *
 *  플레이 단계:
 *    PARENT : 물리/AI + 명령서 생성
 *    CHILD1 : DrawCall 0~1023 GPU 렌더링
 *    CHILD2 : DrawCall 1024~2047 GPU 렌더링
 *
 *  측정: 병렬_md.md 섹션 6.3 출력 형식 준수
 * ============================================================
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sched.h>

/* ── 공통기준.md 고정 상수 ── */
#define FIXED_SEED          0xDEAD4096U
#define MAP_WIDTH           1024
#define MAP_HEIGHT          1024
#define CHUNK_SIZE          64
#define CHUNKS_PER_ROW      (MAP_WIDTH  / CHUNK_SIZE)
#define CHUNKS_PER_COL      (MAP_HEIGHT / CHUNK_SIZE)
#define TOTAL_CHUNKS        (CHUNKS_PER_ROW * CHUNKS_PER_COL)  /* 256 */

#define OCTAVE_COUNT        6
#define MC_SAMPLES          160
#define BLUR_RADIUS         2

#define EROSION_DROPS       ((long)MAP_WIDTH * MAP_HEIGHT * 16)
#define EROSION_STEPS       256

#define MAX_CLIFF_DIFF      0.6f
#define HEIGHT_MIN          0.0f
#define HEIGHT_MAX          1.0f
#define HISTOGRAM_BINS      256

#define LIGHT_SAMPLES       200000000L

#define AI_MONSTERS         20
#define AI_WEIGHTS          16
#define AI_ANNEAL_STEPS     325
#define AI_COMBAT_SIM       500
#define AI_COMBAT_TICKS     300

#define PARTICLE_TYPES      4
#define PARTICLE_COUNT      400
#define PARTICLE_STEPS      2250

#define PLAY_TICKS          500
#define PHYSICS_OBJECTS     5000
#define AI_SIGHT_RAYS       128
#define RENDER_PASSES       8
#define DRAWCALL_COUNT      2048
#define RAYMARCH_STEPS      6000

/* ── Child 수 ── */
#define NUM_CHILDREN        2
#define CHUNKS_PER_CHILD    (TOTAL_CHUNKS / NUM_CHILDREN)   /* 128 */
#define RENDER_PER_CHILD    (DRAWCALL_COUNT / NUM_CHILDREN)  /* 1024 */

/* ─────────────────────────── 유틸 ─────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static uint32_t lcg(uint32_t *s) {
    return (*s = (*s) * 1664525u + 1013904223u);
}
static float lcg_f(uint32_t *s) {
    return (float)(lcg(s) & 0xFFFFFF) / (float)0x1000000;
}
static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

/* ─────────────────────────── CPU snapshot ─────────────────────────── */
typedef struct { long long user,nice,system,idle,iowait,irq,softirq; } CoreStat;
typedef struct { CoreStat cores[16]; int num_cores; } CpuSnapshot;

static void cpu_snapshot(CpuSnapshot *snap) {
    FILE *fp = fopen("/proc/stat","r");
    char line[256];
    snap->num_cores = 0;
    while (fgets(line,sizeof(line),fp)) {
        if (strncmp(line,"cpu",3)!=0) break;
        if (line[3]==' ') continue;
        int idx = atoi(line+3);
        if (idx>=16) continue;
        sscanf(line+3,"%*d %lld %lld %lld %lld %lld %lld %lld",
               &snap->cores[idx].user,&snap->cores[idx].nice,
               &snap->cores[idx].system,&snap->cores[idx].idle,
               &snap->cores[idx].iowait,&snap->cores[idx].irq,
               &snap->cores[idx].softirq);
        snap->num_cores = idx+1;
    }
    fclose(fp);
}
static void cpu_calc_util(const CpuSnapshot *b, const CpuSnapshot *a,
                          double *out, int *nc) {
    *nc = a->num_cores;
    for (int i=0;i<a->num_cores;i++) {
        long long idle  = (a->cores[i].idle+a->cores[i].iowait)
                         -(b->cores[i].idle+b->cores[i].iowait);
        long long total = (a->cores[i].user+a->cores[i].nice+a->cores[i].system
                          +a->cores[i].idle+a->cores[i].iowait
                          +a->cores[i].irq+a->cores[i].softirq)
                         -(b->cores[i].user+b->cores[i].nice+b->cores[i].system
                          +b->cores[i].idle+b->cores[i].iowait
                          +b->cores[i].irq+b->cores[i].softirq);
        out[i] = (total>0)?(1.0-(double)idle/total)*100.0:0.0;
    }
}

typedef struct { long voluntary; long nonvoluntary; } CtxSwitch;
static void read_ctx(CtxSwitch *cs) {
    FILE *fp = fopen("/proc/self/status","r");
    char line[256]; cs->voluntary=0; cs->nonvoluntary=0;
    while (fgets(line,sizeof(line),fp)) {
        if (strncmp(line,"voluntary_ctxt_switches",23)==0)
            sscanf(line,"%*s %ld",&cs->voluntary);
        if (strncmp(line,"nonvoluntary_ctxt_switches",26)==0)
            sscanf(line,"%*s %ld",&cs->nonvoluntary);
    }
    fclose(fp);
}
static long read_rss_kb(void) {
    FILE *fp = fopen("/proc/self/status","r");
    char line[256]; long rss=0;
    while (fgets(line,sizeof(line),fp))
        if (strncmp(line,"VmRSS",5)==0){ sscanf(line,"%*s %ld",&rss); break; }
    fclose(fp); return rss;
}

/* ──────────────────── 지형 구조체 ──────────────────── */
typedef struct {
    float frequency[OCTAVE_COUNT];
    float amplitude[OCTAVE_COUNT];
    float mountain_weight, roughness, persistence, lacunarity;
    uint32_t seed;
} NoiseParams;

typedef struct {
    float height_data[CHUNK_SIZE * CHUNK_SIZE];
} Chunk;

/* 공유 메모리 레이아웃 */
typedef struct {
    float data[MAP_HEIGHT][MAP_WIDTH];         /* 지형맵 4MB */
    NoiseParams params;
    /* Child별 결과 전달용 */
    double child_elapsed_ms[NUM_CHILDREN];
    int    child_integrity[NUM_CHILDREN];
    /* 플레이 단계 렌더링 결과 (공유) */
    float  render_result[DRAWCALL_COUNT];
    double child_render_ms[NUM_CHILDREN];
    /* PARENT 세팅 결과 */
    float  avg_light;
    double parent_light_ms, parent_ai_ms, parent_particle_ms;
    double ai_score;
    float  particle_energy;
} SharedMem;

/* DrawCall */
typedef struct {
    float pos_x, pos_y, pos_z;
    float normal_x, normal_y, normal_z;
    float light_intensity;
    int   pass_mask;
} DrawCall;

/* ──────────────────── Perlin Noise ──────────────────── */
static void init_noise_params(NoiseParams *p) {
    p->seed        = FIXED_SEED;
    p->persistence = 0.5f;
    p->lacunarity  = 2.0f;
    uint32_t rng = p->seed;
    p->mountain_weight = 0.3f + lcg_f(&rng)*0.4f;
    p->roughness       = 0.4f + lcg_f(&rng)*0.4f;
    float freq=1.0f,amp=1.0f,asum=0.0f;
    for (int i=0;i<OCTAVE_COUNT;i++){
        p->frequency[i]=freq*p->roughness;
        p->amplitude[i]=amp;
        asum+=amp; freq*=p->lacunarity; amp*=p->persistence;
    }
    for (int i=0;i<OCTAVE_COUNT;i++) p->amplitude[i]/=asum;
}

static float perlin_octave(float x,float y,float freq,float amp,uint32_t seed){
    float fx=x*freq, fy=y*freq;
    int ix=(int)floorf(fx), iy=(int)floorf(fy);
    float tx=fx-ix, ty=fy-iy;
    float ux=tx*tx*tx*(tx*(tx*6-15)+10);
    float uy=ty*ty*ty*(ty*(ty*6-15)+10);
    uint32_t s00=seed^(uint32_t)(ix*1619+iy*31337);
    uint32_t s10=seed^(uint32_t)((ix+1)*1619+iy*31337);
    uint32_t s01=seed^(uint32_t)(ix*1619+(iy+1)*31337);
    uint32_t s11=seed^(uint32_t)((ix+1)*1619+(iy+1)*31337);
    float g00=sinf((float)s00*1e-5f)*cosf((float)(s00>>8)*1e-5f);
    float g10=sinf((float)s10*1e-5f)*cosf((float)(s10>>8)*1e-5f);
    float g01=sinf((float)s01*1e-5f)*cosf((float)(s01>>8)*1e-5f);
    float g11=sinf((float)s11*1e-5f)*cosf((float)(s11>>8)*1e-5f);
    float v0=g00+ux*(g10-g00), v1=g01+ux*(g11-g01);
    return amp*(v0+uy*(v1-v0));
}

static float octave_stack(float px,float py,const NoiseParams *p){
    float h=0;
    for (int o=0;o<OCTAVE_COUNT;o++)
        h+=perlin_octave(px,py,p->frequency[o],p->amplitude[o],
                         p->seed^(uint32_t)(o*0xDEAD));
    return h;
}

static float monte_carlo(float px,float py,const NoiseParams *p){
    uint32_t rng=p->seed^(uint32_t)(px*7919+py*6271);
    float sum=0,wsum=0;
    for (int s=0;s<MC_SAMPLES;s++){
        float jx=px+(lcg_f(&rng)-0.5f)*0.1f;
        float jy=py+(lcg_f(&rng)-0.5f)*0.1f;
        float dx=jx-px,dy=jy-py;
        float w=expf(-(dx*dx+dy*dy)*8);
        sum+=octave_stack(jx,jy,p)*w; wsum+=w;
    }
    return wsum>0?sum/wsum:0;
}

static void chunk_blur(float *hd){
    static const float K[5][5]={
        {0.00390625f,0.015625f,0.0234375f,0.015625f,0.00390625f},
        {0.015625f,  0.0625f,  0.09375f,  0.0625f,  0.015625f  },
        {0.0234375f, 0.09375f, 0.140625f, 0.09375f, 0.0234375f },
        {0.015625f,  0.0625f,  0.09375f,  0.0625f,  0.015625f  },
        {0.00390625f,0.015625f,0.0234375f,0.015625f,0.00390625f},
    };
    float tmp[CHUNK_SIZE*CHUNK_SIZE];
    memcpy(tmp,hd,sizeof(tmp));
    for (int y=0;y<CHUNK_SIZE;y++)
    for (int x=0;x<CHUNK_SIZE;x++){
        float acc=0;
        for (int ky=-2;ky<=2;ky++)
        for (int kx=-2;kx<=2;kx++){
            int ny=clampf(y+ky,0,CHUNK_SIZE-1);
            int nx=clampf(x+kx,0,CHUNK_SIZE-1);
            acc+=tmp[ny*CHUNK_SIZE+nx]*K[ky+2][kx+2];
        }
        hd[y*CHUNK_SIZE+x]=acc;
    }
}

static void noise_compute_chunk(int chunk_id, float out[CHUNK_SIZE*CHUNK_SIZE],
                                 const NoiseParams *p){
    int cy=chunk_id/CHUNKS_PER_ROW, cx=chunk_id%CHUNKS_PER_ROW;
    for (int ly=0;ly<CHUNK_SIZE;ly++)
    for (int lx=0;lx<CHUNK_SIZE;lx++){
        float px=(cx*CHUNK_SIZE+lx)/(float)MAP_WIDTH;
        float py=(cy*CHUNK_SIZE+ly)/(float)MAP_HEIGHT;
        float h=monte_carlo(px,py,p);
        out[ly*CHUNK_SIZE+lx]=clampf((h+1)*0.5f,HEIGHT_MIN,HEIGHT_MAX);
    }
    chunk_blur(out);
}

/* terrain_merge: chunk 범위 [start, end) → tmap->data */
static void terrain_merge_range(float data[MAP_HEIGHT][MAP_WIDTH],
                                 const NoiseParams *p,
                                 int start_chunk, int end_chunk){
    Chunk *chunks = malloc(sizeof(Chunk)*(end_chunk-start_chunk));
    for (int ci=start_chunk;ci<end_chunk;ci++){
        noise_compute_chunk(ci,chunks[ci-start_chunk].height_data,p);
    }
    for (int ci=start_chunk;ci<end_chunk;ci++){
        int cy=ci/CHUNKS_PER_ROW, cx=ci%CHUNKS_PER_ROW;
        for (int ly=0;ly<CHUNK_SIZE;ly++)
        for (int lx=0;lx<CHUNK_SIZE;lx++)
            data[cy*CHUNK_SIZE+ly][cx*CHUNK_SIZE+lx]
                =chunks[ci-start_chunk].height_data[ly*CHUNK_SIZE+lx];
    }
    free(chunks);
}

/* terrain_blur 전체맵 */
static void terrain_blur(float data[MAP_HEIGHT][MAP_WIDTH]){
    float (*map)[MAP_WIDTH] = data;
    const int ITER = 400;
    float *tmp = malloc(sizeof(float)*MAP_HEIGHT*MAP_WIDTH);
    if (!tmp) return;
    for (int iter = 0; iter < ITER; iter++) {
        for (int y = 0; y < MAP_HEIGHT; y++)
            memcpy(tmp+y*MAP_WIDTH, map[y], sizeof(float)*MAP_WIDTH);
        for (int cy = 1; cy < CHUNKS_PER_COL; cy++) {
            int gy = cy*CHUNK_SIZE;
            for (int brow = gy-BLUR_RADIUS; brow <= gy+BLUR_RADIUS; brow++) {
                if (brow<0||brow>=MAP_HEIGHT) continue;
                for (int x = 0; x < MAP_WIDTH; x++) {
                    float ls=0,lss=0;
                    for (int dy=-1;dy<=1;dy++) {
                        int ny=brow+dy;
                        if (ny<0||ny>=MAP_HEIGHT) continue;
                        float v=tmp[ny*MAP_WIDTH+x]; ls+=v; lss+=v*v;
                    }
                    float var=(lss/3.0f)-(ls/3.0f)*(ls/3.0f);
                    float w=1.0f/(1.0f+expf(-var*10.0f));
                    float acc=0; int cnt=0;
                    for (int dy=-BLUR_RADIUS;dy<=BLUR_RADIUS;dy++) {
                        int ny=brow+dy;
                        if (ny<0||ny>=MAP_HEIGHT) continue;
                        acc+=tmp[ny*MAP_WIDTH+x]*(1-w)+tmp[brow*MAP_WIDTH+x]*w;
                        cnt++;
                    }
                    map[brow][x]=acc/(float)cnt;
                }
            }
        }
        for (int y = 0; y < MAP_HEIGHT; y++)
            memcpy(tmp+y*MAP_WIDTH, map[y], sizeof(float)*MAP_WIDTH);
        for (int cx = 1; cx < CHUNKS_PER_ROW; cx++) {
            int gx = cx*CHUNK_SIZE;
            for (int bcol = gx-BLUR_RADIUS; bcol <= gx+BLUR_RADIUS; bcol++) {
                if (bcol<0||bcol>=MAP_WIDTH) continue;
                for (int y = 0; y < MAP_HEIGHT; y++) {
                    float ls=0,lss=0;
                    for (int dx=-1;dx<=1;dx++) {
                        int nx=bcol+dx;
                        if (nx<0||nx>=MAP_WIDTH) continue;
                        float v=tmp[y*MAP_WIDTH+nx]; ls+=v; lss+=v*v;
                    }
                    float var=(lss/3.0f)-(ls/3.0f)*(ls/3.0f);
                    float w=1.0f/(1.0f+expf(-var*10.0f));
                    float acc=0; int cnt=0;
                    for (int dx=-BLUR_RADIUS;dx<=BLUR_RADIUS;dx++) {
                        int nx=bcol+dx;
                        if (nx<0||nx>=MAP_WIDTH) continue;
                        acc+=tmp[y*MAP_WIDTH+nx]*(1-w)+tmp[y*MAP_WIDTH+bcol]*w;
                        cnt++;
                    }
                    map[y][bcol]=acc/(float)cnt;
                }
            }
        }
    }
    free(tmp);
}

static void terrain_erosion(float data[MAP_HEIGHT][MAP_WIDTH]){
    const float er=0.004f,dr=0.002f;
    uint32_t rng=FIXED_SEED^0xBAADF00D;
    for (long d=0;d<EROSION_DROPS;d++){
        int fx=((rng=rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_WIDTH-2)+1;
        int fy=((rng=rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_HEIGHT-2)+1;
        float hc=data[fy][fx];
        float hm=hc; int dx=0,dy=0;
        if(data[fy][fx+1]<hm){hm=data[fy][fx+1];dx=1;dy=0;}
        if(data[fy][fx-1]<hm){hm=data[fy][fx-1];dx=-1;dy=0;}
        if(data[fy+1][fx]<hm){hm=data[fy+1][fx];dx=0;dy=1;}
        if(data[fy-1][fx]<hm){hm=data[fy-1][fx];dx=0;dy=-1;}
        if(dx==0&&dy==0) continue;
        float diff=hc-hm;
        data[fy][fx]    =clampf(data[fy][fx]   -er*diff,HEIGHT_MIN,HEIGHT_MAX);
        data[fy+dy][fx+dx]=clampf(data[fy+dy][fx+dx]+dr*diff,HEIGHT_MIN,HEIGHT_MAX);
    }
}

static void terrain_histogram(float data[MAP_HEIGHT][MAP_WIDTH]){
    long hist[HISTOGRAM_BINS]={0};
    for (int y=0;y<MAP_HEIGHT;y++)
    for (int x=0;x<MAP_WIDTH;x++){
        int b=(int)(data[y][x]*(HISTOGRAM_BINS-1));
        b=b<0?0:b>HISTOGRAM_BINS-1?HISTOGRAM_BINS-1:b;
        hist[b]++;
    }
    long tot=MAP_WIDTH*MAP_HEIGHT;
    float cdf[HISTOGRAM_BINS]; long cum=0; long cmin=-1;
    for (int b=0;b<HISTOGRAM_BINS;b++){
        cum+=hist[b];
        if(cmin<0&&hist[b]>0) cmin=cum-hist[b];
        cdf[b]=(float)cum;
    }
    for (int b=0;b<HISTOGRAM_BINS;b++){
        cdf[b]=clampf((cdf[b]-(float)cmin)/(float)(tot-cmin),0,1);
    }
    for (int y=0;y<MAP_HEIGHT;y++)
    for (int x=0;x<MAP_WIDTH;x++){
        int b=(int)(data[y][x]*(HISTOGRAM_BINS-1));
        b=b<0?0:b>HISTOGRAM_BINS-1?HISTOGRAM_BINS-1:b;
        data[y][x]=cdf[b];
    }
}

static int terrain_integrity(const float data[MAP_HEIGHT][MAP_WIDTH]){
    int fail=0;
    for (int y=0;y<MAP_HEIGHT;y++)
    for (int x=0;x<MAP_WIDTH;x++){
        float h=data[y][x];
        if(h<HEIGHT_MIN-1e-4f||h>HEIGHT_MAX+1e-4f) fail++;
        if(x+1<MAP_WIDTH&&fabsf(h-data[y][x+1])>MAX_CLIFF_DIFF) fail++;
        if(y+1<MAP_HEIGHT&&fabsf(h-data[y+1][x])>MAX_CLIFF_DIFF) fail++;
    }
    return fail;
}

/* ──────────────────── 조도 계산 ──────────────────── */
static float parent_lighting(void){
    uint32_t rng=FIXED_SEED^0xABCD1234;
    double total=0;
    float sun_az =lcg_f(&rng)*2.0f*3.14159f;
    float sun_alt=0.2f+lcg_f(&rng)*0.6f;
    float sun_x=cosf(sun_alt)*cosf(sun_az);
    float sun_y=sinf(sun_alt);
    float sun_z=cosf(sun_alt)*sinf(sun_az);
    for (long s=0;s<LIGHT_SAMPLES;s++){
        float u=lcg_f(&rng),v=lcg_f(&rng);
        float theta=acosf(sqrtf(u)), phi=2.0f*3.14159f*v;
        float rx=sinf(theta)*cosf(phi);
        float ry=cosf(theta);
        float rz=sinf(theta)*sinf(phi);
        float direct=rx*sun_x+ry*sun_y+rz*sun_z;
        if(direct<0) direct=0;
        float sx=lcg_f(&rng),sy=lcg_f(&rng);
        float scatter=expf(-sx*sx*2.0f-sy*sy*2.0f)*0.3f;
        total+=direct+scatter;
    }
    return (float)(total/LIGHT_SAMPLES);
}

/* ──────────────────── 몬스터 AI ──────────────────── */
static float parent_monster_ai(void){
    uint32_t rng=FIXED_SEED^0xDEADBEEF;
    float global_best=0.0f;
    for (int m=0;m<AI_MONSTERS;m++){
        float w[AI_WEIGHTS],best_w[AI_WEIGHTS];
        for (int i=0;i<AI_WEIGHTS;i++) w[i]=best_w[i]=lcg_f(&rng)*2-1;
        float best_score=-1e9f;
        float T=1.0f;
        const float cool=1.0f-(1.0f-1e-4f)/AI_ANNEAL_STEPS;
        for (int st=0;st<AI_ANNEAL_STEPS;st++){
            float cand[AI_WEIGHTS];
            for (int i=0;i<AI_WEIGHTS;i++)
                cand[i]=w[i]+(lcg_f(&rng)-0.5f)*T*0.5f;
            float matk=fabsf(cand[0])*10+5;
            float aggr=clampf(cand[1]*0.5f+0.5f,0,1);
            float evad=clampf(cand[2]*0.5f+0.5f,0,1);
            double score_sum=0;
            for (int sim=0;sim<AI_COMBAT_SIM;sim++){
                float mhp=100,php=100;
                float atk_s=cand[3]*(mhp/100.0f)+cand[4]*aggr;
                float ret_s=cand[5]*(1-(mhp/100.0f))+cand[6]*evad;
                for (int tk=0;tk<AI_COMBAT_TICKS;tk++){
                    if(mhp<=0||php<=0) break;
                    float hr=mhp/100.0f;
                    atk_s=cand[3]*hr+cand[4]*aggr;
                    ret_s=cand[5]*(1-hr)+cand[6]*evad;
                    if(atk_s>ret_s){
                        float dmg=matk*(1+cand[7]*0.2f);
                        for (int k=0;k<AI_WEIGHTS;k++) dmg+=cand[k]*cand[k]*0.01f;
                        php-=dmg*0.016f;
                        mhp-=8*(1-evad*0.3f)*0.016f;
                    } else {
                        float ev=evad*(1+sinf((float)tk*0.1f)*0.2f);
                        mhp-=8*(1-ev)*0.016f;
                    }
                    mhp=clampf(mhp,0,100);
                    php=clampf(php,0,100);
                }
                score_sum+=(100-php)*0.6f+mhp*0.4f;
            }
            float score=(float)(score_sum/AI_COMBAT_SIM);
            float delta=score-best_score;
            if(delta>0||lcg_f(&rng)<expf(delta/T)){
                memcpy(w,cand,sizeof(w));
                if(score>best_score){ best_score=score; memcpy(best_w,w,sizeof(w));}
            }
            T*=cool;
        }
        if(best_score>global_best) global_best=best_score;
    }
    return global_best;
}

/* ──────────────────── 파티클 시뮬 ──────────────────── */
typedef enum { EXPLOSION=0,SPLASH,LAVA,SMOKE } EffectType;
typedef struct { float x,y,z,vx,vy,vz,fx,fy,fz,mass,charge,lifetime; } Particle;

static float simulate_effect(EffectType type, uint32_t seed){
    float gravity,viscosity,init_speed,eps,sigma;
    int lt_max;
    switch(type){
        case EXPLOSION: gravity=9.8f;viscosity=0.01f;init_speed=15;eps=0.5f;sigma=0.3f;lt_max=2;break;
        case SPLASH:    gravity=9.8f;viscosity=0.1f; init_speed=5; eps=1.0f;sigma=0.5f;lt_max=4;break;
        case LAVA:      gravity=9.8f;viscosity=2.0f; init_speed=2; eps=2.0f;sigma=0.8f;lt_max=8;break;
        default:        gravity=-1.0f;viscosity=0.05f;init_speed=1;eps=0.1f;sigma=1.0f;lt_max=12;break;
    }
    uint32_t rng=seed^(uint32_t)(type*0x1234);
    Particle *p=malloc(sizeof(Particle)*PARTICLE_COUNT);
    for (int i=0;i<PARTICLE_COUNT;i++){
        p[i].x=(lcg_f(&rng)-0.5f)*2;
        p[i].y=(lcg_f(&rng)-0.5f)*2;
        p[i].z=(lcg_f(&rng)-0.5f)*2;
        float spd=init_speed*(0.5f+lcg_f(&rng)*0.5f);
        p[i].vx=(lcg_f(&rng)-0.5f)*spd;
        p[i].vy=(lcg_f(&rng)-0.5f)*spd;
        p[i].vz=(lcg_f(&rng)-0.5f)*spd;
        p[i].mass=0.5f+lcg_f(&rng)*0.5f;
        p[i].charge=lcg_f(&rng)*2-1;
        p[i].lifetime=lt_max*(0.5f+lcg_f(&rng)*0.5f);
        p[i].fx=p[i].fy=p[i].fz=0;
    }
    double total_ke=0;
    const float DT=0.001f;
    for (int st=0;st<PARTICLE_STEPS;st++){
        for (int i=0;i<PARTICLE_COUNT;i++) p[i].fx=p[i].fy=p[i].fz=0;
        for (int i=0;i<PARTICLE_COUNT;i++){
            if(p[i].lifetime<=0) continue;
            for (int j=i+1;j<PARTICLE_COUNT;j++){
                if(p[j].lifetime<=0) continue;
                float dx=p[j].x-p[i].x,dy=p[j].y-p[i].y,dz=p[j].z-p[i].z;
                float r2=dx*dx+dy*dy+dz*dz;
                if(r2<0.01f) continue;
                float sr=sigma/sqrtf(r2);
                float sr6=sr*sr*sr*sr*sr*sr;
                float sr12=sr6*sr6;
                float fmag=24*eps*(2*sr12-sr6)/r2;
                if(fmag>1e5f)fmag=1e5f; if(fmag<-1e5f)fmag=-1e5f;
                p[i].fx-=fmag*dx; p[i].fy-=fmag*dy; p[i].fz-=fmag*dz;
                p[j].fx+=fmag*dx; p[j].fy+=fmag*dy; p[j].fz+=fmag*dz;
            }
        }
        for (int i=0;i<PARTICLE_COUNT;i++){
            if(p[i].lifetime<=0) continue;
            p[i].fy+=gravity*p[i].mass;
            p[i].fx-=viscosity*p[i].vx;
            p[i].fy-=viscosity*p[i].vy;
            p[i].fz-=viscosity*p[i].vz;
            if(type==SMOKE){
                p[i].fx+=0.5f*sinf(p[i].y*3+(float)st*DT*2);
                p[i].fz+=0.5f*cosf(p[i].x*3+(float)st*DT*2);
            }
            p[i].vx+=p[i].fx/p[i].mass*DT;
            p[i].vy+=p[i].fy/p[i].mass*DT;
            p[i].vz+=p[i].fz/p[i].mass*DT;
            if(p[i].vx>1e4f)p[i].vx=1e4f; if(p[i].vx<-1e4f)p[i].vx=-1e4f;
            if(p[i].vy>1e4f)p[i].vy=1e4f; if(p[i].vy<-1e4f)p[i].vy=-1e4f;
            if(p[i].vz>1e4f)p[i].vz=1e4f; if(p[i].vz<-1e4f)p[i].vz=-1e4f;
            p[i].x+=p[i].vx*DT; p[i].y+=p[i].vy*DT; p[i].z+=p[i].vz*DT;
            total_ke+=0.5*p[i].mass*(p[i].vx*p[i].vx+p[i].vy*p[i].vy+p[i].vz*p[i].vz);
            p[i].lifetime-=DT;
        }
    }
    free(p);
    return (float)(total_ke/(PARTICLE_STEPS*(double)PARTICLE_COUNT));
}

static float parent_particle_sim(void){
    float total=0;
    for (int t=0;t<PARTICLE_TYPES;t++)
        total+=simulate_effect((EffectType)t, FIXED_SEED^(uint32_t)(t*0x5678));
    return total / PARTICLE_TYPES;
}

/* ──────────────────── 플레이 단계 ──────────────────── */
static void tick_physics_ai(uint32_t *rng, const float data[MAP_HEIGHT][MAP_WIDTH]){
    for (int o=0;o<PHYSICS_OBJECTS;o++){
        float px=lcg_f(rng)*MAP_WIDTH, py=lcg_f(rng)*MAP_HEIGHT;
        float vx=(lcg_f(rng)-0.5f)*2.0f, vy=(lcg_f(rng)-0.5f)*2.0f;
        int hits=0;
        for (int r=0;r<AI_SIGHT_RAYS;r++){
            float angle=lcg_f(rng)*6.2832f, dist=1.0f+lcg_f(rng)*64.0f;
            float tx=px+cosf(angle)*dist, ty=py+sinf(angle)*dist;
            int mx=(int)clampf(tx,0,MAP_WIDTH-1), mz=(int)clampf(ty,0,MAP_HEIGHT-1);
            float h=data[(int)clampf(py,0,MAP_HEIGHT-1)][(int)clampf(px,0,MAP_WIDTH-1)];
            float th=data[mz][mx];
            if(th>h+0.1f) hits++;
        }
        float h=data[(int)clampf(py,0,MAP_HEIGHT-1)][(int)clampf(px,0,MAP_WIDTH-1)];
        vy+=(9.8f*h-0.1f*vy)*0.016f; vx+=(-0.1f*vx)*0.016f;
        px+=vx*0.016f; py+=vy*0.016f;
        (void)hits;
    }
}

static void tick_build_commands(DrawCall *cmds, uint32_t *rng,
                                const float data[MAP_HEIGHT][MAP_WIDTH], int tick){
    for (int i=0;i<DRAWCALL_COUNT;i++){
        float px=lcg_f(rng)*(MAP_WIDTH-2);
        float py=lcg_f(rng)*(MAP_HEIGHT-2);
        int ix=(int)px, iy=(int)py;
        float dh_dx=data[iy][ix+1]-data[iy][ix];
        float dh_dy=data[iy+1][ix]-data[iy][ix];
        float len=sqrtf(dh_dx*dh_dx+dh_dy*dh_dy+1.0f);
        cmds[i].pos_x=px; cmds[i].pos_y=data[iy][ix]*64; cmds[i].pos_z=py;
        cmds[i].normal_x=-dh_dx/len;
        cmds[i].normal_y=1.0f/len;
        cmds[i].normal_z=-dh_dy/len;
        int hits=0;
        for (int r=0;r<AI_SIGHT_RAYS;r++){
            float ang=lcg_f(rng)*6.2832f,dist=1.0f+lcg_f(rng)*64.0f;
            int mx=(int)clampf(px+cosf(ang)*dist,0,MAP_WIDTH-1);
            int mz=(int)clampf(py+sinf(ang)*dist,0,MAP_HEIGHT-1);
            if(data[mz][mx]>data[iy][ix]+0.1f) hits++;
        }
        cmds[i].light_intensity=clampf((float)hits/(float)(PHYSICS_OBJECTS*AI_SIGHT_RAYS),0,1);
        cmds[i].pass_mask=1<<(i%RENDER_PASSES);
        (void)tick;
    }
}

/* 렌더링: start~end DrawCall 처리 */
static float tick_render_range(const DrawCall *cmds,
                               const float data[MAP_HEIGHT][MAP_WIDTH],
                               int start, int end, int tick){
    float pixel_sum=0;
    for (int i=start;i<end;i++){
        float ray_ox=512.0f,ray_oy=2.0f,ray_oz=512.0f;
        float ray_dx=cmds[i].pos_x-ray_ox;
        float ray_dy=cmds[i].pos_y-ray_oy;
        float ray_dz=cmds[i].pos_z-ray_oz;
        float ray_len=sqrtf(ray_dx*ray_dx+ray_dy*ray_dy+ray_dz*ray_dz);
        if(ray_len<1e-6f) continue;
        ray_dx/=ray_len; ray_dy/=ray_len; ray_dz/=ray_len;
        float occlusion=0;
        for (int s=0;s<RAYMARCH_STEPS;s++){
            float t=ray_len*(float)s/RAYMARCH_STEPS;
            float sx=ray_ox+ray_dx*t;
            float sy=ray_oy+ray_dy*t;
            float sz=ray_oz+ray_dz*t;
            int mx=(int)clampf(sx,0,MAP_WIDTH-1);
            int mz=(int)clampf(sz,0,MAP_HEIGHT-1);
            float terrain_h=data[mz][mx];
            if(sy<terrain_h) occlusion+=1.0f;
        }
        occlusion/=RAYMARCH_STEPS;
        float pixel=(1.0f-occlusion)*cmds[i].light_intensity
                   *(1.0f+sinf((float)tick*0.1f+cmds[i].pos_x*0.01f)*0.05f);
        pixel_sum+=pixel;
    }
    return pixel_sum;
}

/* ══════════════════════════════════════════════
   CHILD 함수: 세팅 단계 지형 생성 (청크 분할)
   ══════════════════════════════════════════════ */
static void child_terrain_setup(SharedMem *shm, int child_id){
    int start_chunk = child_id * CHUNKS_PER_CHILD;
    int end_chunk   = start_chunk + CHUNKS_PER_CHILD;
    double t0 = now_ms();
    printf("[CHILD%d] 지형 noise 시작: 청크 %d~%d\n",
           child_id+1, start_chunk, end_chunk-1);
    terrain_merge_range(shm->data, &shm->params, start_chunk, end_chunk);
    printf("[CHILD%d] noise 완료: %.2f ms\n", child_id+1, now_ms()-t0);
    shm->child_elapsed_ms[child_id] = now_ms()-t0;
    /* integrity는 통합 blur/erosion 후 Parent가 검사 */
}

/* ══════════════════════════════════════════════
   CHILD 함수: 플레이 단계 렌더링 분할
   ══════════════════════════════════════════════ */
static void child_render_play(SharedMem *shm, const DrawCall *cmds,
                               int child_id, int tick){
    int start = child_id * RENDER_PER_CHILD;
    int end   = start + RENDER_PER_CHILD;
    double t0=now_ms();
    float r = tick_render_range(cmds, shm->data, start, end, tick);
    shm->render_result[child_id] = r;
    shm->child_render_ms[child_id] += now_ms()-t0;
}

/* ══════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════ */
int main(void){
    /* 공유 메모리 할당 */
    size_t shm_sz = sizeof(SharedMem);
    SharedMem *shm = mmap(NULL,shm_sz,PROT_READ|PROT_WRITE,
                          MAP_SHARED|MAP_ANONYMOUS,-1,0);
    memset(shm,0,sizeof(*shm));

    /* DrawCall 공유 메모리 */
    size_t dc_sz = sizeof(DrawCall)*DRAWCALL_COUNT;
    DrawCall *cmds = mmap(NULL,dc_sz,PROT_READ|PROT_WRITE,
                          MAP_SHARED|MAP_ANONYMOUS,-1,0);

    /* 코어 수 확인 */
    int num_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    printf("=== 다 실험: PARENT1 + CHILD%d 병렬 ===\n", NUM_CHILDREN);
    printf("시드: 0x%X | 맵: %dx%d | 청크: %d개\n",
           FIXED_SEED,MAP_WIDTH,MAP_HEIGHT,TOTAL_CHUNKS);
    printf("[CORE] 시스템 코어 수: %d\n", num_cores);

    /* Parent → cpu0 고정 */
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(0,&cpuset);
    sched_setaffinity(0,sizeof(cpuset),&cpuset);
    printf("[PIN] PARENT → cpu0 / OS 예약 코어 없음\n\n");

    /* 노이즈 파라미터 초기화 */
    init_noise_params(&shm->params);

    /* ────────────── 세팅 단계 ────────────── */
    CpuSnapshot snap_setup_b, snap_setup_a;
    CtxSwitch ctx_setup_b, ctx_setup_a;
    cpu_snapshot(&snap_setup_b);
    read_ctx(&ctx_setup_b);
    double t_setup_start = now_ms();

    /* fork Child 들 */
    double t_fork[NUM_CHILDREN];
    pid_t  pids[NUM_CHILDREN];
    for (int c=0;c<NUM_CHILDREN;c++){
        double tf=now_ms();
        pid_t pid=fork();
        t_fork[c]=now_ms()-tf;
        if(pid==0){
            /* CHILD: 코어 고정 (cpu1, cpu2) */
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(c+1,&cs);
            sched_setaffinity(0,sizeof(cs),&cs);
            child_terrain_setup(shm,c);
            exit(0);
        }
        pids[c]=pid;
    }

    /* PARENT: 조도+AI+파티클 병렬 실행 */
    printf("[PARENT] 조도 계산 시작 (200M 샘플)...\n");
    double tl=now_ms();
    shm->avg_light = parent_lighting();
    shm->parent_light_ms = now_ms()-tl;
    printf("[PARENT] 조도 완료: %.6f | %.2f ms\n",
           shm->avg_light, shm->parent_light_ms);

    printf("[PARENT] 몬스터 AI 시작...\n");
    double ta=now_ms();
    shm->ai_score = parent_monster_ai();
    shm->parent_ai_ms = now_ms()-ta;
    printf("[PARENT] AI 완료: %.4f | %.2f ms\n",
           shm->ai_score, shm->parent_ai_ms);

    printf("[PARENT] 파티클 시뮬레이션 시작...\n");
    double tp=now_ms();
    shm->particle_energy = parent_particle_sim();
    shm->parent_particle_ms = now_ms()-tp;
    printf("[PARENT] 파티클 완료: %.2f ms\n", shm->parent_particle_ms);

    /* Child 완료 대기 */
    double t_wait = now_ms();
    for (int c=0;c<NUM_CHILDREN;c++) waitpid(pids[c],NULL,0);
    double wait_ms = now_ms()-t_wait;

    /* BARRIER POINT 1: 모든 noise 완료 → merge 확인 후 blur 시작 */
    printf("[PARENT] BARRIER1: 모든 Child noise 완료. blur 시작\n");
    double tb=now_ms();
    terrain_blur(shm->data);
    double blur_ms=now_ms()-tb;
    printf("[PARENT] 블러 완료: %.2f ms\n",blur_ms);

    /* BARRIER POINT 2 */
    double te=now_ms();
    terrain_erosion(shm->data);
    double erosion_ms=now_ms()-te;
    printf("[PARENT] 침식 완료: %.2f ms\n",erosion_ms);

    /* BARRIER POINT 3 */
    double th_t=now_ms();
    terrain_histogram(shm->data);
    double hist_ms=now_ms()-th_t;
    int integrity_fails=terrain_integrity(shm->data);
    printf("[PARENT] 히스토그램+무결성 완료: %.2f ms | 실패: %d\n",
           hist_ms, integrity_fails);

    double wall_setup = now_ms()-t_setup_start;
    cpu_snapshot(&snap_setup_a);
    read_ctx(&ctx_setup_a);

    printf("\n[세팅 완료] 벽시계: %.2f ms\n\n", wall_setup);

    /* ────────────── 플레이 단계 ────────────── */
    CpuSnapshot snap_play_b, snap_play_a;
    CtxSwitch ctx_play_b, ctx_play_a;
    cpu_snapshot(&snap_play_b);
    read_ctx(&ctx_play_b);
    double t_play_start = now_ms();

    uint32_t rng = FIXED_SEED^0xCAFEF00D;
    double phys_ms=0, cmd_ms=0;
    memset(shm->child_render_ms,0,sizeof(shm->child_render_ms));

    printf("[PLAY] 시작: %d틱 | PARENT 물리/명령서 + CHILD렌더링 분할\n",PLAY_TICKS);
    for (int tick=0;tick<PLAY_TICKS;tick++){
        /* PARENT: 물리/AI */
        double tp0=now_ms();
        tick_physics_ai(&rng, shm->data);
        phys_ms+=now_ms()-tp0;

        /* PARENT: 명령서 */
        double tc0=now_ms();
        tick_build_commands(cmds,&rng,shm->data,tick);
        cmd_ms+=now_ms()-tc0;

        /* fork Child들 → 렌더링 분할 */
        for (int c=0;c<NUM_CHILDREN;c++){
            pid_t pid=fork();
            if(pid==0){
                cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(c+1,&cs);
                sched_setaffinity(0,sizeof(cs),&cs);
                child_render_play(shm,cmds,c,tick);
                exit(0);
            }
            pids[c]=pid;
        }
        for (int c=0;c<NUM_CHILDREN;c++) waitpid(pids[c],NULL,0);

        if((tick+1)%50==0){
            printf("[PLAY] 틱 %3d/%d | 물리: %.1fms | 명령서: %.1fms | 렌더[C1+C2]: %.1f+%.1fms\n",
                   tick+1,PLAY_TICKS,phys_ms,cmd_ms,
                   shm->child_render_ms[0],shm->child_render_ms[1]);
        }
    }

    double wall_play = now_ms()-t_play_start;
    cpu_snapshot(&snap_play_a);
    read_ctx(&ctx_play_a);
    printf("[PLAY] 완료: %.2f ms\n\n", wall_play);

    /* ────────────── 성능 지표 출력 ────────────── */
    double total_wall = wall_setup + wall_play;

    /* CPU user+sys */
    struct rusage ru;
    getrusage(RUSAGE_SELF,&ru);
    double cpu_user=ru.ru_utime.tv_sec*1000.0+ru.ru_utime.tv_usec/1000.0;
    double cpu_sys =ru.ru_stime.tv_sec*1000.0+ru.ru_stime.tv_usec/1000.0;
    double cpu_util=(cpu_user+cpu_sys)/total_wall*100.0;

    /* 코어별 활용률 */
    double setup_util[16],play_util[16]; int nc;
    cpu_calc_util(&snap_setup_b,&snap_setup_a,setup_util,&nc);
    cpu_calc_util(&snap_play_b, &snap_play_a, play_util, &nc);

    /* Speedup / Efficiency
       세팅 N: Parent 계산 참여 + Child2 = 3
       플레이 N: Parent(물리+명령서) + Child2(렌더) = 3
       통합 N = 3 */
    int N_setting = 3;  /* Parent(조도+AI+파티클) + Child1 + Child2 */
    int N_play    = 3;  /* Parent(물리+명령서) + Child1(렌더) + Child2(렌더) */
    double baseline_ms = 111683.74; /* 본인 Baseline 측정값 */
    double team_baseline_ms = 361601.0;
    double speedup    = baseline_ms / total_wall;
    double efficiency = speedup / N_setting * 100.0;
    double P_eff = (1.0-1.0/speedup)/(1.0-(1.0/N_setting));
    double S_theory = 1.0/((1.0-P_eff)+P_eff/N_setting);

    /* 순차 환산 (세팅) */
    double seq_setup = shm->child_elapsed_ms[0]+shm->child_elapsed_ms[1]
                      +shm->parent_light_ms+shm->parent_ai_ms+shm->parent_particle_ms;
    double setup_speedup = seq_setup/wall_setup;

    printf("===========================================================\n");
    printf("     다 실험: PARENT1 + CHILD%d 병렬\n", NUM_CHILDREN);
    printf("===========================================================\n");
    printf("[실험 구성]\n");
    printf("  Worker 유형: Child\n");
    printf("  Worker 수 K: %d\n", NUM_CHILDREN);
    printf("  실행 주체  : Parent 1 + Child %d\n", NUM_CHILDREN);
    printf("  active core N (세팅): %d  ← Parent 계산+Child2\n", N_setting);
    printf("  active core N (플레이): %d  ← Parent+Child2 렌더\n", N_play);
    printf("  N 근거: Parent가 조도/AI/파티클 실제 계산 참여 → N에 포함\n");
    printf("\n");

    printf("----- [CHILD×%d / TERRAIN] 세팅 지형 연산 -----\n", NUM_CHILDREN);
    printf("  [CHILD1] noise 청크 0~127:   %.2f ms\n", shm->child_elapsed_ms[0]);
    printf("  [CHILD2] noise 청크 128~255: %.2f ms\n", shm->child_elapsed_ms[1]);
    printf("  블러:                         %.2f ms  <- BARRIER POINT 1\n", blur_ms);
    printf("  침식:                         %.2f ms  <- BARRIER POINT 2\n", erosion_ms);
    printf("  히스토그램/무결성:            %.2f ms  <- BARRIER POINT 3\n", hist_ms);
    printf("------------------------------------------------------\n");
    printf("  [PARENT] 조도 계산:          %.2f ms  (Child 병렬 중)\n", shm->parent_light_ms);
    printf("  [PARENT] 몬스터AI:           %.2f ms\n", shm->parent_ai_ms);
    printf("  [PARENT] 파티클:             %.2f ms\n", shm->parent_particle_ms);
    printf("  세팅 벽시계(병렬):           %.2f ms\n", wall_setup);
    printf("  세팅 순차 환산:              %.2f ms\n", seq_setup);
    printf("  세팅 Speedup:                    %.2fx\n", setup_speedup);
    printf("  세팅 Efficiency:                 %.1f%%  (= Speedup / %d)\n",
           setup_speedup/N_setting*100.0, N_setting);
    printf("\n");
    printf("----- [PLAY] %d틱 -----\n", PLAY_TICKS);
    printf("  [PARENT] 물리/AI:            %.2f ms  (틱평균 %.2f ms)\n",
           phys_ms, phys_ms/PLAY_TICKS);
    printf("  [PARENT] 명령서 생성:        %.2f ms  (틱평균 %.2f ms)\n",
           cmd_ms, cmd_ms/PLAY_TICKS);
    printf("  [CHILD1] GPU 렌더0~1023:     %.2f ms  (틱평균 %.2f ms)\n",
           shm->child_render_ms[0], shm->child_render_ms[0]/PLAY_TICKS);
    printf("  [CHILD2] GPU 렌더1024~2047:  %.2f ms  (틱평균 %.2f ms)\n",
           shm->child_render_ms[1], shm->child_render_ms[1]/PLAY_TICKS);
    printf("  플레이 벽시계:               %.2f ms\n", wall_play);
    printf("\n");

    printf("===========================================================\n");
    printf("  총 소요 시간:                %.2f ms\n", total_wall);
    printf("  Speedup: %.2fx  (Baseline: %.2f ms — 본인 측정값)\n",
           speedup, baseline_ms);
    printf("  [참고] 팀 기준 Baseline: %.0f ms → Speedup: %.2fx\n",
           team_baseline_ms, team_baseline_ms/total_wall);
    printf("  Efficiency: %.1f%%  (= Speedup / N=%d)\n", efficiency, N_setting);
    printf("===========================================================\n");
    printf("[Amdahl's Law 검증]\n");
    printf("  active core 수 N:            %d\n", N_setting);
    printf("  병렬화 비율 P_effective:     %.1f%%\n", P_eff*100.0);
    printf("  이론 Speedup:                %.2fx\n", S_theory);
    printf("  실측 Speedup:                %.2fx\n", speedup);
    printf("  오차:                        %.2fx\n", S_theory-speedup);
    printf("===========================================================\n");
    printf("[CPU 성능 지표]\n");
    printf("  CPU user: %.2f ms | sys: %.2f ms\n", cpu_user, cpu_sys);
    printf("  CPU 활용률: %.1f%%\n", cpu_util);
    printf("===========================================================\n");
    printf("[세팅 단계 코어별]        [플레이 단계 코어별]\n");
    for (int i=0;i<nc;i++){
        const char *slbl="", *plbl="";
        if(i==0){slbl="(PARENT)";plbl="(PARENT)";}
        else if(i==1){slbl="(CHILD1)";plbl="(CHILD1-렌더)";}
        else if(i==2){slbl="(CHILD2)";plbl="(CHILD2-렌더)";}
        printf("  cpu%2d%-12s: %5.1f%%          cpu%2d%-12s: %5.1f%%\n",
               i,slbl,setup_util[i],i,plbl,play_util[i]);
    }
    printf("===========================================================\n");
    printf("[컨텍스트 스위칭]\n");
    printf("               voluntary    nonvoluntary\n");
    printf("  세팅 단계: %6ld 회    %8ld 회\n",
           ctx_setup_a.voluntary-ctx_setup_b.voluntary,
           ctx_setup_a.nonvoluntary-ctx_setup_b.nonvoluntary);
    printf("  플레이 단계: %6ld 회  %8ld 회\n",
           ctx_play_a.voluntary-ctx_play_b.voluntary,
           ctx_play_a.nonvoluntary-ctx_play_b.nonvoluntary);
    printf("===========================================================\n");
    printf("[IPC 오버헤드]\n");
    printf("  지형맵: 공유메모리 직접 참조 (pipe IPC 없음, 0 bytes)\n");
    printf("  렌더결과: 공유메모리 직접 참조 (0 bytes)\n");
    printf("  세팅/플레이 waitpid: %.3f ms / %.3f ms\n", wait_ms, 0.0);
    printf("===========================================================\n");
    printf("[프로세스 생성]\n");
    printf("  fork()×%d (세팅): %.3f ms / %.3f ms\n",
           NUM_CHILDREN, t_fork[0], t_fork[1]);
    printf("  fork()×%d×%d틱 (플레이): 틱당 Child fork 비용 포함\n",
           NUM_CHILDREN, PLAY_TICKS);
    printf("===========================================================\n");
    printf("[메모리 사용량]\n");
    printf("  현재 RSS: %ld KB\n", read_rss_kb());
    printf("===========================================================\n");
    printf("[결과 검증 — Baseline과 동일해야 함]\n");
    printf("  평균 조도:     %.6f  (기준: 0.397698) %s\n",
           shm->avg_light, fabsf(shm->avg_light-0.397698f)<1e-4f?"✓":"✗");
    printf("  AI 최고점수:   %.4f  (기준: 97.7063)  %s\n",
           shm->ai_score, fabsf(shm->ai_score-97.7063f)<0.1f?"✓":"✗");
    printf("  파티클 에너지: %.6f  (기준: 25324.066406) %s\n",
           shm->particle_energy,
           fabsf(shm->particle_energy-25324.066406f)<1.0f?"✓":"✗");
    printf("  무결성 실패:   %d    (기준: 0) %s\n",
           integrity_fails, integrity_fails==0?"✓":"✗");
    printf("===========================================================\n");

    munmap(shm,shm_sz);
    munmap(cmds,dc_sz);
    return 0;
}
