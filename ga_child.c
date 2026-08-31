/*
 * ============================================================
 *  game_world_ga_exp.c
 *  가 실험: PARENT1 + CHILD1 병렬
 * ============================================================
 *
 *  병렬 구조:
 *    [세팅 단계]
 *      공유메모리에 tmap 할당 → fork()
 *      CHILD:  지형맵 생성 (노이즈→병합→블러→침식→히스토그램) → tmap->data(공유메모리)
 *      PARENT: 조도계산 + 몬스터AI + 파티클 (독립 연산)
 *      PARENT waitpid → CHILD 종료 확인 후 공유메모리 포인터로 플레이 단계 진행
 *
 *    [플레이 단계]
 *      PARENT: 물리/AI 연산 + 명령서 생성 → cmds 공유메모리 기록 → pipe 신호
 *      CHILD:  pipe 신호 수신 → tick_render() 전담 → 결과 pipe 반환
 *      tmap->data는 공유메모리 직접 참조 (복사 없음)
 *
 *  측정 지표 (measurement_guide.md 기반):
 *    - fork/waitpid 시간
 *    - pipe IPC 오버헤드 (write/read 시간, 전송량)
 *    - 세팅/플레이 단계 분리 코어별 CPU 활용률
 *    - 컨텍스트 스위칭 (세팅/플레이 분리)
 *    - Speedup / Efficiency (Baseline 대비)
 *    - 메모리 사용량
 *    - 결과 검증값 4개
 *
 *  빌드:
 *    gcc -O2 -o game_world_ga_exp game_world_ga_exp.c -lm
 * ============================================================
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <sys/resource.h>
#include <sched.h>

/* ============================================================
 * 섹션 0: 상수
 * ============================================================ */

#define FIXED_SEED      0xDEAD4096U

/* Baseline Wall Time (5회 중앙값으로 확정 후 교체) */
#define BASELINE_WALL_MS  111683.74

#define MAP_WIDTH       1024
#define MAP_HEIGHT      1024
#define CHUNK_SIZE      64
#define CHUNKS_PER_ROW  (MAP_WIDTH  / CHUNK_SIZE)
#define CHUNKS_PER_COL  (MAP_HEIGHT / CHUNK_SIZE)
#define TOTAL_CHUNKS    (CHUNKS_PER_ROW * CHUNKS_PER_COL)

#define OCTAVE_COUNT    6
#define MC_SAMPLES      160
#define BLUR_RADIUS     2

#define EROSION_DROPS   ((long)MAP_WIDTH * MAP_HEIGHT * 16)
#define EROSION_STEPS   256

#define MAX_CLIFF_DIFF  0.6f
#define HEIGHT_MIN      0.0f
#define HEIGHT_MAX      1.0f
#define HISTOGRAM_BINS  256

#define LIGHT_SAMPLES   200000000L

#define AI_MONSTERS     20
#define AI_WEIGHTS      16
#define AI_ANNEAL_STEPS 325
#define AI_COMBAT_SIM   500
#define AI_COMBAT_TICKS 300

#define PARTICLE_TYPES  4
#define PARTICLE_COUNT  400
#define PARTICLE_STEPS  2250

#define PLAY_TICKS          500
#define PHYSICS_OBJECTS     5000
#define AI_SIGHT_RAYS       128
#define RENDER_PASSES       8
#define DRAWCALL_COUNT      2048
#define RAYMARCH_STEPS      6000

/* 코어 고정 (병렬 실험) */
#define CORE_PARENT  0
#define CORE_CHILD   1

/* DrawCall pipe 전송 크기 */
#define DRAWCALL_BYTES  ((int)(sizeof(DrawCall) * DRAWCALL_COUNT))

/* ============================================================
 * 섹션 1: 자료구조
 * ============================================================ */

typedef struct {
    uint32_t seed;
    float    frequency[OCTAVE_COUNT];
    float    amplitude[OCTAVE_COUNT];
    float    persistence, lacunarity;
    float    mountain_weight, roughness;
} NoiseParams;

typedef struct {
    int    chunk_id, chunk_x, chunk_y;
    int    pixel_x_start, pixel_y_start;
    float *height_data;
} ChunkDesc;

typedef struct {
    float     data[MAP_HEIGHT][MAP_WIDTH];
    NoiseParams params;
    ChunkDesc   chunks[TOTAL_CHUNKS];
    double    terrain_elapsed_ms;  /* CHILD가 기록 → PARENT가 waitpid 후 읽음 */
    int       integrity_fails;
} TerrainMap;

typedef struct {
    float x, y, z, vx, vy, vz, fx, fy, fz;
    float mass, charge, lifetime;
} Particle;

typedef struct {
    int   object_id;
    float pos_x, pos_y, pos_z;
    float normal_x, normal_y, normal_z;
    float light_intensity;
    int   pass_mask;
} DrawCall;

typedef struct {
    double pixel_sum;
    double render_ms;
} TickResult;

/* ── 측정 지표 구조체 ── */
typedef struct {
    long long user, nice, system, idle, iowait, irq, softirq;
} CoreStat;
typedef struct { CoreStat cores[16]; int num_cores; } CpuSnapshot;
typedef struct { long voluntary; long nonvoluntary; } CtxSwitch;

typedef struct {
    /* 세팅 단계 */
    double setting_wall_ms;
    double terrain_ms;      /* CHILD 지형 생성 */
    double lighting_ms;     /* PARENT 조도 */
    double ai_ms;           /* PARENT AI */
    double particle_ms;     /* PARENT 파티클 */
    /* 플레이 단계 */
    double play_wall_ms;
    double play_physics_ms;
    double play_cmd_ms;
    double play_render_ms;
    /* 전체 */
    double total_ms;
    /* 생성/종료 오버헤드 */
    double fork_ms;
    double waitpid_setting_ms;
    double waitpid_play_ms;
    /* IPC 오버헤드 */
    double pipe_write_setting_ms;  /* 세팅 결과 전송 */
    double pipe_read_setting_ms;
    long   pipe_bytes_setting;
    double pipe_write_play_ms;     /* 플레이 명령서 전송 합계 */
    double pipe_read_play_ms;
    long   pipe_bytes_play;
    double pipe_write_result_ms;   /* 플레이 결과 반환 합계 */
    double pipe_read_result_ms;
    /* CPU 지표 */
    double cpu_user_ms;
    double cpu_sys_ms;
    double setup_core_util[16];
    double play_core_util[16];
    int    num_cores;
    /* 컨텍스트 스위칭 */
    long   setup_vol_ctx, setup_nonvol_ctx;
    long   play_vol_ctx,  play_nonvol_ctx;
    /* 메모리 */
    long   mem_rss_setup_kb, mem_peak_setup_kb;
    long   mem_rss_play_kb,  mem_peak_play_kb;
    /* 검증값 */
    int    integrity_fails;
    float  lighting_result;
    float  ai_best_score;
    float  particle_final_energy;
    double pixel_grand;
} PerfMetrics;

/* ============================================================
 * 섹션 2: 측정 유틸리티
 * ============================================================ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static uint32_t lcg(uint32_t *s) { *s=(*s)*1664525u+1013904223u; return *s; }
static float lcg_f(uint32_t *s) { return (float)(lcg(s)&0xFFFFFF)/(float)0x1000000; }
static float clampf(float v,float lo,float hi){return v<lo?lo:v>hi?hi:v;}

static void pin_to_core(int core_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        printf("[WARN] cpu%d 고정 실패\n", core_id);
}

static void cpu_snapshot(CpuSnapshot *snap) {
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return;
    char line[256];
    snap->num_cores = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "cpu", 3) != 0) break;
        if (line[3] == ' ') continue;
        int idx = atoi(line + 3);
        if (idx >= 16) continue;
        sscanf(line + 3, "%*d %lld %lld %lld %lld %lld %lld %lld",
               &snap->cores[idx].user,   &snap->cores[idx].nice,
               &snap->cores[idx].system, &snap->cores[idx].idle,
               &snap->cores[idx].iowait, &snap->cores[idx].irq,
               &snap->cores[idx].softirq);
        snap->num_cores = idx + 1;
    }
    fclose(fp);
}

static void cpu_calc_util(const CpuSnapshot *before, const CpuSnapshot *after,
                           double *util_out, int *num_cores_out) {
    *num_cores_out = after->num_cores;
    for (int i = 0; i < after->num_cores; i++) {
        long long idle_diff =
            (after->cores[i].idle  + after->cores[i].iowait) -
            (before->cores[i].idle + before->cores[i].iowait);
        long long total_diff =
            (after->cores[i].user   + after->cores[i].nice   +
             after->cores[i].system + after->cores[i].idle   +
             after->cores[i].iowait + after->cores[i].irq    +
             after->cores[i].softirq) -
            (before->cores[i].user   + before->cores[i].nice   +
             before->cores[i].system + before->cores[i].idle   +
             before->cores[i].iowait + before->cores[i].irq    +
             before->cores[i].softirq);
        util_out[i] = (total_diff > 0) ?
            (1.0 - (double)idle_diff / total_diff) * 100.0 : 0.0;
    }
}

static void read_ctx_switches(CtxSwitch *cs) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return;
    char line[256];
    cs->voluntary = 0; cs->nonvoluntary = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "voluntary_ctxt_switches", 23) == 0)
            sscanf(line, "%*s %ld", &cs->voluntary);
        if (strncmp(line, "nonvoluntary_ctxt_switches", 26) == 0)
            sscanf(line, "%*s %ld", &cs->nonvoluntary);
    }
    fclose(fp);
}

static void read_memory_usage(long *rss_kb, long *peak_kb) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return;
    char line[256];
    *rss_kb = 0; *peak_kb = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmRSS:", 6) == 0)
            sscanf(line, "%*s %ld", rss_kb);
        if (strncmp(line, "VmPeak:", 7) == 0)
            sscanf(line, "%*s %ld", peak_kb);
    }
    fclose(fp);
}

/* pipe 전체 쓰기/읽기 (EINTR 처리) */
static ssize_t write_all(int fd, const void *buf, size_t n) {
    size_t written = 0;
    while (written < n) {
        ssize_t r = write(fd, (const char*)buf + written, n - written);
        if (r <= 0) return -1;
        written += (size_t)r;
    }
    return (ssize_t)written;
}
static ssize_t read_all(int fd, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char*)buf + got, n - got);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return (ssize_t)got;
}

/* ============================================================
 * 섹션 3: 지형 생성 함수들 (baseline과 동일)
 * ============================================================ */

static float perlin_octave(float x, float y, float freq, float amp, uint32_t seed) {
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

static float octave_stack(float px, float py, const NoiseParams *p) {
    float h=0;
    for (int oct=0;oct<OCTAVE_COUNT;oct++)
        h+=perlin_octave(px,py,p->frequency[oct],p->amplitude[oct],
                         p->seed^(uint32_t)(oct*0xDEAD));
    return h;
}

static float monte_carlo(float px, float py, const NoiseParams *p) {
    uint32_t rng=p->seed^(uint32_t)(px*7919+py*6271);
    float sum=0, wsum=0;
    for (int s=0;s<MC_SAMPLES;s++) {
        float jx=px+(lcg_f(&rng)-0.5f)*0.1f;
        float jy=py+(lcg_f(&rng)-0.5f)*0.1f;
        float h=octave_stack(jx,jy,p);
        float dx=jx-px, dy=jy-py;
        float w=expf(-(dx*dx+dy*dy)*8);
        sum+=h*w; wsum+=w;
    }
    return wsum>0?sum/wsum:0;
}

static void chunk_blur(float *data, int w, int h) {
    static const float K[5][5]={
        {0.00390625f,0.015625f,0.0234375f,0.015625f,0.00390625f},
        {0.015625f,0.0625f,0.09375f,0.0625f,0.015625f},
        {0.0234375f,0.09375f,0.140625f,0.09375f,0.0234375f},
        {0.015625f,0.0625f,0.09375f,0.0625f,0.015625f},
        {0.00390625f,0.015625f,0.0234375f,0.015625f,0.00390625f},
    };
    float *tmp=malloc(sizeof(float)*(size_t)(w*h));
    if(!tmp)return;
    memcpy(tmp,data,sizeof(float)*(size_t)(w*h));
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        float acc=0,ws=0;
        for(int ky=-2;ky<=2;ky++) for(int kx=-2;kx<=2;kx++){
            int nx=x+kx,ny=y+ky;
            if(nx<0||nx>=w||ny<0||ny>=h)continue;
            float kv=K[ky+2][kx+2];
            acc+=tmp[ny*w+nx]*kv; ws+=kv;
        }
        data[y*w+x]=ws>0?acc/ws:tmp[y*w+x];
    }
    free(tmp);
}

static void noise_compute_chunk(ChunkDesc *c, const NoiseParams *p) {
    for(int ly=0;ly<CHUNK_SIZE;ly++) for(int lx=0;lx<CHUNK_SIZE;lx++){
        float px=(float)(c->pixel_x_start+lx)/MAP_WIDTH;
        float py=(float)(c->pixel_y_start+ly)/MAP_HEIGHT;
        float h=monte_carlo(px,py,p);
        c->height_data[ly*CHUNK_SIZE+lx]=clampf((h+1)*0.5f,HEIGHT_MIN,HEIGHT_MAX);
    }
    chunk_blur(c->height_data,CHUNK_SIZE,CHUNK_SIZE);
}

static void terrain_merge(TerrainMap *tmap) {
    for(int ci=0;ci<TOTAL_CHUNKS;ci++){
        ChunkDesc *c=&tmap->chunks[ci];
        for(int ly=0;ly<CHUNK_SIZE;ly++){
            int gy=c->pixel_y_start+ly;
            memcpy(&tmap->data[gy][c->pixel_x_start],
                   &c->height_data[ly*CHUNK_SIZE],sizeof(float)*CHUNK_SIZE);
        }
    }
}

static void terrain_blur(TerrainMap *tmap) {
    float (*map)[MAP_WIDTH]=tmap->data;
    const int ITER=400;
    float *tmp=malloc(sizeof(float)*MAP_HEIGHT*MAP_WIDTH);
    if(!tmp)return;
    for(int iter=0;iter<ITER;iter++){
        for(int y=0;y<MAP_HEIGHT;y++)
            memcpy(tmp+y*MAP_WIDTH,map[y],sizeof(float)*MAP_WIDTH);
        for(int cy=1;cy<CHUNKS_PER_COL;cy++){
            int gy=cy*CHUNK_SIZE;
            for(int brow=gy-BLUR_RADIUS;brow<=gy+BLUR_RADIUS;brow++){
                if(brow<0||brow>=MAP_HEIGHT)continue;
                for(int x=0;x<MAP_WIDTH;x++){
                    float ls=0,lss=0;
                    for(int dy=-1;dy<=1;dy++){
                        int ny=brow+dy;
                        if(ny<0||ny>=MAP_HEIGHT)continue;
                        float v=tmp[ny*MAP_WIDTH+x];ls+=v;lss+=v*v;
                    }
                    float var=(lss/3.0f)-(ls/3.0f)*(ls/3.0f);
                    float w=1.0f/(1.0f+expf(-var*10.0f));
                    float acc=0;int cnt=0;
                    for(int dy=-BLUR_RADIUS;dy<=BLUR_RADIUS;dy++){
                        int ny=brow+dy;
                        if(ny<0||ny>=MAP_HEIGHT)continue;
                        acc+=tmp[ny*MAP_WIDTH+x]*(1-w)+tmp[brow*MAP_WIDTH+x]*w;
                        cnt++;
                    }
                    map[brow][x]=acc/(float)cnt;
                }
            }
        }
        for(int y=0;y<MAP_HEIGHT;y++)
            memcpy(tmp+y*MAP_WIDTH,map[y],sizeof(float)*MAP_WIDTH);
        for(int cx=1;cx<CHUNKS_PER_ROW;cx++){
            int gx=cx*CHUNK_SIZE;
            for(int bcol=gx-BLUR_RADIUS;bcol<=gx+BLUR_RADIUS;bcol++){
                if(bcol<0||bcol>=MAP_WIDTH)continue;
                for(int y=0;y<MAP_HEIGHT;y++){
                    float ls=0,lss=0;
                    for(int dx=-1;dx<=1;dx++){
                        int nx=bcol+dx;
                        if(nx<0||nx>=MAP_WIDTH)continue;
                        float v=tmp[y*MAP_WIDTH+nx];ls+=v;lss+=v*v;
                    }
                    float var=(lss/3.0f)-(ls/3.0f)*(ls/3.0f);
                    float w=1.0f/(1.0f+expf(-var*10.0f));
                    float acc=0;int cnt=0;
                    for(int dx=-BLUR_RADIUS;dx<=BLUR_RADIUS;dx++){
                        int nx=bcol+dx;
                        if(nx<0||nx>=MAP_WIDTH)continue;
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

static void terrain_erosion(TerrainMap *tmap) {
    float (*map)[MAP_WIDTH]=tmap->data;
    const float er=0.004f,dr=0.002f;
    uint32_t rng=tmap->params.seed^0xBAADF00D;
    for(long d=0;d<EROSION_DROPS;d++){
        int fx=(int)(((rng=rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_WIDTH-2))+1;
        int fy=(int)(((rng=rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_HEIGHT-2))+1;
        for(int step=0;step<EROSION_STEPS;step++){
            if(fx<1||fx>=MAP_WIDTH-1||fy<1||fy>=MAP_HEIGHT-1)break;
            float hc=map[fy][fx],hm=hc;int dx=0,dy=0;
            if(map[fy][fx+1]<hm){hm=map[fy][fx+1];dx=1;dy=0;}
            if(map[fy][fx-1]<hm){hm=map[fy][fx-1];dx=-1;dy=0;}
            if(map[fy+1][fx]<hm){hm=map[fy+1][fx];dx=0;dy=1;}
            if(map[fy-1][fx]<hm){hm=map[fy-1][fx];dx=0;dy=-1;}
            if(!dx&&!dy)break;
            float diff=hc-hm;
            map[fy][fx]-=er*diff;map[fy+dy][fx+dx]+=dr*diff;
            fx+=dx;fy+=dy;
        }
    }
    for(int y=0;y<MAP_HEIGHT;y++) for(int x=0;x<MAP_WIDTH;x++)
        map[y][x]=clampf(map[y][x],HEIGHT_MIN,HEIGHT_MAX);
}

static void terrain_histogram(TerrainMap *tmap) {
    float (*map)[MAP_WIDTH]=tmap->data;
    uint32_t hist[HISTOGRAM_BINS]={0};
    for(int y=0;y<MAP_HEIGHT;y++) for(int x=0;x<MAP_WIDTH;x++){
        int b=(int)(map[y][x]*(HISTOGRAM_BINS-1));
        b=b<0?0:b>=HISTOGRAM_BINS?HISTOGRAM_BINS-1:b;hist[b]++;
    }
    float cdf[HISTOGRAM_BINS];uint32_t cum=0,cmin=0;int found=0;
    for(int b=0;b<HISTOGRAM_BINS;b++){
        cum+=hist[b];cdf[b]=(float)cum;
        if(!found&&hist[b]>0){cmin=cum;found=1;}
    }
    uint32_t tot=MAP_WIDTH*MAP_HEIGHT;
    for(int b=0;b<HISTOGRAM_BINS;b++){
        cdf[b]=(cdf[b]-(float)cmin)/(float)(tot-cmin);
        cdf[b]=clampf(cdf[b],0,1);
    }
    for(int y=0;y<MAP_HEIGHT;y++) for(int x=0;x<MAP_WIDTH;x++){
        int b=(int)(map[y][x]*(HISTOGRAM_BINS-1));
        b=b<0?0:b>=HISTOGRAM_BINS?HISTOGRAM_BINS-1:b;map[y][x]=cdf[b];
    }
}

static int terrain_integrity(const TerrainMap *tmap) {
    int fail=0;
    for(int y=0;y<MAP_HEIGHT;y++) for(int x=0;x<MAP_WIDTH;x++){
        float h=tmap->data[y][x];
        if(h<HEIGHT_MIN-1e-4f||h>HEIGHT_MAX+1e-4f){fail++;continue;}
        if(x+1<MAP_WIDTH&&fabsf(h-tmap->data[y][x+1])>MAX_CLIFF_DIFF)fail++;
        if(y+1<MAP_HEIGHT&&fabsf(h-tmap->data[y+1][x])>MAX_CLIFF_DIFF)fail++;
    }
    return fail;
}

static void init_noise_params(NoiseParams *p) {
    p->seed=FIXED_SEED;p->persistence=0.5f;p->lacunarity=2.0f;
    uint32_t rng=FIXED_SEED;
    p->mountain_weight=0.3f+lcg_f(&rng)*0.4f;
    p->roughness=0.4f+lcg_f(&rng)*0.4f;
    float freq=1,amp=1,asum=0;
    for(int i=0;i<OCTAVE_COUNT;i++){
        p->frequency[i]=freq*p->roughness;p->amplitude[i]=amp;
        asum+=amp;freq*=p->lacunarity;amp*=p->persistence;
    }
    for(int i=0;i<OCTAVE_COUNT;i++) p->amplitude[i]/=asum;
    printf("[CHILD] Seed: 0x%08X | Mountain: %.2f | Roughness: %.2f\n",
           p->seed,p->mountain_weight,p->roughness);
}

/* ============================================================
 * 섹션 4: CHILD 담당 연산 (조도 + AI + 파티클)
 * ============================================================ */

static float lighting_simulation(uint32_t seed) {
    printf("[PARENT] 조도 계산 시작 (%ldM 샘플)...\n",LIGHT_SAMPLES/1000000L);
    uint32_t rng=seed^0xABCD1234;
    double total=0.0;
    float sun_az=lcg_f(&rng)*2.0f*3.14159f;
    float sun_alt=0.2f+lcg_f(&rng)*0.6f;
    float sun_x=cosf(sun_alt)*cosf(sun_az);
    float sun_y=sinf(sun_alt);
    float sun_z=cosf(sun_alt)*sinf(sun_az);
    for(long i=0;i<LIGHT_SAMPLES;i++){
        float sx=lcg_f(&rng),sy=lcg_f(&rng);
        float u=lcg_f(&rng),v=lcg_f(&rng);
        float theta=acosf(sqrtf(u));
        float phi=2.0f*3.14159f*v;
        float rx=sinf(theta)*cosf(phi);
        float ry=cosf(theta);
        float rz=sinf(theta)*sinf(phi);
        float direct=rx*sun_x+ry*sun_y+rz*sun_z;
        if(direct<0)direct=0;
        float scatter=expf(-sx*sx*2.0f-sy*sy*2.0f)*0.3f;
        total+=(double)(direct+scatter);
    }
    float result=(float)(total/LIGHT_SAMPLES);
    printf("[PARENT] 조도 완료: %.6f\n",result);
    return result;
}

static float monster_ai(uint32_t seed) {
    printf("[PARENT] 몬스터 AI 시작...\n");
    uint32_t rng=seed^0xDEADBEEF;
    float global_best=0.0f;
    for(int m=0;m<AI_MONSTERS;m++){
        float w[AI_WEIGHTS],best_w[AI_WEIGHTS];
        for(int i=0;i<AI_WEIGHTS;i++) w[i]=best_w[i]=lcg_f(&rng)*2-1;
        float best_score=-1e9f;
        float T=1.0f;
        const float cool=1.0f-(1.0f-1e-4f)/AI_ANNEAL_STEPS;
        for(int step=0;step<AI_ANNEAL_STEPS;step++){
            float cand[AI_WEIGHTS];
            for(int i=0;i<AI_WEIGHTS;i++)
                cand[i]=w[i]+(lcg_f(&rng)-0.5f)*T*0.5f;
            double score_sum=0;
            for(int sim=0;sim<AI_COMBAT_SIM;sim++){
                float mhp=100,php=100;
                float matk=fabsf(cand[0])*10+5;
                float aggr=clampf(cand[1]*0.5f+0.5f,0,1);
                float evad=clampf(cand[2]*0.5f+0.5f,0,1);
                for(int tick=0;tick<AI_COMBAT_TICKS;tick++){
                    if(mhp<=0||php<=0)break;
                    float hr=mhp/100.0f;
                    float atk_s=cand[3]*hr+cand[4]*aggr;
                    float ret_s=cand[5]*(1-hr)+cand[6]*evad;
                    if(atk_s>ret_s){
                        float dmg=matk*(1+cand[7]*0.2f);
                        for(int k=0;k<AI_WEIGHTS;k++) dmg+=cand[k]*cand[k]*0.01f;
                        php-=dmg*0.016f;
                        mhp-=8*(1-evad*0.3f)*0.016f;
                    } else {
                        float ev=evad*(1+sinf((float)tick*0.1f)*0.2f);
                        mhp-=8*(1-ev)*0.016f;
                    }
                    mhp=clampf(mhp,0,100);php=clampf(php,0,100);
                }
                score_sum+=(100-php)*0.6f+mhp*0.4f;
            }
            float score=(float)(score_sum/AI_COMBAT_SIM);
            float delta=score-best_score;
            if(delta>0||lcg_f(&rng)<expf(delta/T)){
                memcpy(w,cand,sizeof(w));
                if(score>best_score){best_score=score;memcpy(best_w,w,sizeof(w));}
            }
            T*=cool;
        }
        if(best_score>global_best)global_best=best_score;
    }
    printf("[PARENT] AI 완료: %.4f\n",global_best);
    return global_best;
}

typedef enum{EXPLOSION=0,SPLASH,LAVA,SMOKE}EffectType;

static float simulate_effect(EffectType type, uint32_t seed){
    const char *names[]={"폭발","물보라","용암","연기"};
    float gravity,viscosity,init_speed,eps,sigma,lt_max;
    switch(type){
        case EXPLOSION:gravity=9.8f;viscosity=0.01f;init_speed=15;eps=0.5f;sigma=0.3f;lt_max=2;break;
        case SPLASH:gravity=9.8f;viscosity=0.1f;init_speed=5;eps=1.0f;sigma=0.5f;lt_max=4;break;
        case LAVA:gravity=9.8f;viscosity=2.0f;init_speed=2;eps=2.0f;sigma=0.8f;lt_max=8;break;
        case SMOKE:gravity=-1.f;viscosity=0.05f;init_speed=1;eps=0.1f;sigma=1.0f;lt_max=12;break;
        default:gravity=9.8f;viscosity=0.1f;init_speed=5;eps=1.0f;sigma=0.5f;lt_max=4;break;
    }
    printf("[PARENT] %s 시뮬레이션 (%d개×%d스텝)...\n",names[type],PARTICLE_COUNT,PARTICLE_STEPS);
    Particle *p=malloc(sizeof(Particle)*PARTICLE_COUNT);
    if(!p)return 0;
    uint32_t rng=seed^(uint32_t)(type*0x1234);
    for(int i=0;i<PARTICLE_COUNT;i++){
        p[i].x=(lcg_f(&rng)-0.5f)*2;p[i].y=(lcg_f(&rng)-0.5f)*2;p[i].z=(lcg_f(&rng)-0.5f)*2;
        float spd=init_speed*(0.5f+lcg_f(&rng)*0.5f);
        p[i].vx=(lcg_f(&rng)-0.5f)*spd;p[i].vy=(lcg_f(&rng)-0.5f)*spd;p[i].vz=(lcg_f(&rng)-0.5f)*spd;
        p[i].fx=p[i].fy=p[i].fz=0;
        p[i].mass=0.5f+lcg_f(&rng)*0.5f;
        p[i].charge=lcg_f(&rng)*2-1;
        p[i].lifetime=lt_max*(0.5f+lcg_f(&rng)*0.5f);
    }
    const float DT=0.001f;
    double total_ke=0;
    for(int step=0;step<PARTICLE_STEPS;step++){
        for(int i=0;i<PARTICLE_COUNT;i++) p[i].fx=p[i].fy=p[i].fz=0;
        for(int i=0;i<PARTICLE_COUNT;i++){
            if(p[i].lifetime<=0)continue;
            for(int j=i+1;j<PARTICLE_COUNT;j++){
                if(p[j].lifetime<=0)continue;
                float dx=p[j].x-p[i].x,dy=p[j].y-p[i].y,dz=p[j].z-p[i].z;
                float r2=dx*dx+dy*dy+dz*dz;
                if(r2<0.01f)continue;
                float r=sqrtf(r2);
                float sr=sigma/r,sr6=sr*sr*sr*sr*sr*sr,sr12=sr6*sr6;
                float fmag=24*eps*(2*sr12-sr6)/r2;
                if(fmag>1e5f)fmag=1e5f;if(fmag<-1e5f)fmag=-1e5f;
                p[i].fx-=fmag*dx;p[i].fy-=fmag*dy;p[i].fz-=fmag*dz;
                p[j].fx+=fmag*dx;p[j].fy+=fmag*dy;p[j].fz+=fmag*dz;
            }
        }
        for(int i=0;i<PARTICLE_COUNT;i++){
            if(p[i].lifetime<=0)continue;
            p[i].fy+=gravity*p[i].mass;
            p[i].fx-=viscosity*p[i].vx;p[i].fy-=viscosity*p[i].vy;p[i].fz-=viscosity*p[i].vz;
            if(type==SMOKE){
                p[i].fx+=0.5f*sinf(p[i].y*3+(float)step*DT*2);
                p[i].fz+=0.5f*cosf(p[i].x*3+(float)step*DT*2);
            }
            p[i].vx+=p[i].fx/p[i].mass*DT;p[i].vy+=p[i].fy/p[i].mass*DT;p[i].vz+=p[i].fz/p[i].mass*DT;
            if(p[i].vx>1e4f)p[i].vx=1e4f;if(p[i].vx<-1e4f)p[i].vx=-1e4f;
            if(p[i].vy>1e4f)p[i].vy=1e4f;if(p[i].vy<-1e4f)p[i].vy=-1e4f;
            if(p[i].vz>1e4f)p[i].vz=1e4f;if(p[i].vz<-1e4f)p[i].vz=-1e4f;
            p[i].x+=p[i].vx*DT;p[i].y+=p[i].vy*DT;p[i].z+=p[i].vz*DT;
            p[i].lifetime-=DT;
            total_ke+=0.5*p[i].mass*(p[i].vx*p[i].vx+p[i].vy*p[i].vy+p[i].vz*p[i].vz);
        }
    }
    float result=(float)(total_ke/(PARTICLE_STEPS*PARTICLE_COUNT));
    printf("[PARENT] %s 완료. 에너지: %.6f\n",names[type],result);
    free(p);
    return result;
}

static float particle_simulation(uint32_t seed){
    float total=0;
    for(int t=0;t<PARTICLE_TYPES;t++)
        total+=simulate_effect((EffectType)t,seed^(uint32_t)(t*0x5678));
    return total/PARTICLE_TYPES;
}

/* ============================================================
 * 섹션 5: 플레이 단계 연산 함수
 * ============================================================ */

static int tick_physics_ai(const TerrainMap *tmap, uint32_t *rng, int tick) {
    int hits=0;(void)tick;
    for(int obj=0;obj<PHYSICS_OBJECTS;obj++){
        float px=lcg_f(rng)*MAP_WIDTH;
        float py=lcg_f(rng)*MAP_HEIGHT;
        float vx=(lcg_f(rng)-0.5f)*2.0f;
        float vy=(lcg_f(rng)-0.5f)*2.0f;
        float h=tmap->data[(int)clampf(py,0,MAP_HEIGHT-1)]
                          [(int)clampf(px,0,MAP_WIDTH-1)];
        vy+=(9.8f*h-0.1f*vy)*0.016f;
        vx+=(-0.1f*vx)*0.016f;
        px+=vx*0.016f;py+=vy*0.016f;
        px=clampf(px,0,MAP_WIDTH-1);py=clampf(py,0,MAP_HEIGHT-1);
        for(int r=0;r<AI_SIGHT_RAYS;r++){
            float angle=lcg_f(rng)*6.2832f;
            float dist=1.0f+lcg_f(rng)*64.0f;
            float tx=px+cosf(angle)*dist;
            float ty=py+sinf(angle)*dist;
            int ix=(int)clampf(tx,0,MAP_WIDTH-1);
            int iy=(int)clampf(ty,0,MAP_HEIGHT-1);
            float th=tmap->data[iy][ix];
            if(th>h+0.1f)hits++;
        }
    }
    return hits;
}

static void tick_build_commands(const TerrainMap *tmap, DrawCall *cmds,
                                 int hits, uint32_t *rng) {
    for(int i=0;i<DRAWCALL_COUNT;i++){
        float px=lcg_f(rng)*(MAP_WIDTH-2);
        float py=lcg_f(rng)*(MAP_HEIGHT-2);
        int ix=(int)px,iy=(int)py;
        float dh_dx=tmap->data[iy][ix+1]-tmap->data[iy][ix];
        float dh_dy=tmap->data[iy+1][ix]-tmap->data[iy][ix];
        float len=sqrtf(dh_dx*dh_dx+dh_dy*dh_dy+1.0f);
        cmds[i].object_id=i;
        cmds[i].pos_x=px;cmds[i].pos_y=tmap->data[iy][ix];cmds[i].pos_z=py;
        cmds[i].normal_x=-dh_dx/len;cmds[i].normal_y=1.0f/len;cmds[i].normal_z=-dh_dy/len;
        cmds[i].light_intensity=clampf(
            (float)hits/(float)(PHYSICS_OBJECTS*AI_SIGHT_RAYS),0.0f,1.0f);
        cmds[i].pass_mask=1<<(i%RENDER_PASSES);
    }
}

/* CHILD가 실행하는 렌더링 — tmap은 공유메모리에서 직접 읽기 */
static double tick_render(const DrawCall *cmds, const TerrainMap *tmap, int tick) {
    double pixel_sum=0.0;
    double *frame_acc=calloc(DRAWCALL_COUNT,sizeof(double));
    if(!frame_acc)return 0;
    for(int pass=0;pass<RENDER_PASSES;pass++){
        for(int i=0;i<DRAWCALL_COUNT;i++){
            if(!(cmds[i].pass_mask&(1<<pass)))continue;
            float ray_ox=512.0f,ray_oy=2.0f,ray_oz=512.0f;
            float ray_dx=cmds[i].pos_x-ray_ox;
            float ray_dy=cmds[i].pos_y-ray_oy;
            float ray_dz=cmds[i].pos_z-ray_oz;
            float ray_len=sqrtf(ray_dx*ray_dx+ray_dy*ray_dy+ray_dz*ray_dz);
            if(ray_len<1e-6f)continue;
            ray_dx/=ray_len;ray_dy/=ray_len;ray_dz/=ray_len;
            float occlusion=0.0f;
            for(int s=0;s<RAYMARCH_STEPS;s++){
                float t=ray_len*(float)s/RAYMARCH_STEPS;
                float sx=ray_ox+ray_dx*t;
                float sy=ray_oy+ray_dy*t;
                float sz=ray_oz+ray_dz*t;
                int mx=(int)clampf(sx,0,MAP_WIDTH-1);
                int mz=(int)clampf(sz,0,MAP_HEIGHT-1);
                if(sy<tmap->data[mz][mx])occlusion+=1.0f;
            }
            occlusion/=RAYMARCH_STEPS;
            float pixel=(1.0f-occlusion)*cmds[i].light_intensity
                *(1.0f+sinf((float)tick*0.1f+cmds[i].pos_x*0.01f)*0.05f);
            frame_acc[i]+=(double)pixel;
            pixel_sum+=(double)pixel;
        }
    }
    free(frame_acc);
    return pixel_sum;
}

/* ============================================================
 * 섹션 6: 성능 출력
 * ============================================================ */

static void print_ascii_map(const TerrainMap *tmap){
    printf("\n[ASCII 미니맵] 64x32:\n");
    const char *sh=" .:;+=xX$&#";
    for(int y=0;y<32;y++){
        for(int x=0;x<64;x++){
            float h=tmap->data[y*MAP_HEIGHT/32][x*MAP_WIDTH/64];
            int li=(int)(h*9);li=li<0?0:li>9?9:li;
            putchar(sh[li]);
        }
        putchar('\n');
    }
}

static void print_metrics(const PerfMetrics *m) {
    double speedup    = BASELINE_WALL_MS / m->total_ms;
    double efficiency = speedup / 2.0 * 100.0;  /* PARENT + CHILD = 2코어 */
    double cpu_total  = m->cpu_user_ms + m->cpu_sys_ms;
    double cpu_util   = (m->total_ms > 0) ? cpu_total / m->total_ms * 100.0 : 0;

    printf("\n===========================================================\n");
    printf("     가 실험: PARENT1 + CHILD1 병렬\n");
    printf("===========================================================\n");
    printf("[세팅 단계]\n");
    printf("  CHILD  지형 생성:      %8.2f ms\n", m->terrain_ms);
    printf("  PARENT 조도:           %8.2f ms\n", m->lighting_ms);
    printf("  PARENT 몬스터AI:       %8.2f ms\n", m->ai_ms);
    printf("  PARENT 파티클:         %8.2f ms\n", m->particle_ms);
    printf("  세팅 벽시계(병렬):     %8.2f ms\n", m->setting_wall_ms);
    printf("  세팅 순차 환산:        %8.2f ms\n",
           m->terrain_ms + m->lighting_ms + m->ai_ms + m->particle_ms);
    printf("  세팅 Speedup:          %8.2fx\n",
           (m->terrain_ms + m->lighting_ms + m->ai_ms + m->particle_ms)
           / m->setting_wall_ms);
    printf("===========================================================\n");
    printf("[플레이 단계] %d틱\n", PLAY_TICKS);
    printf("  PARENT 물리/AI:        %8.2f ms  (틱평균 %.2f ms)\n",
           m->play_physics_ms, m->play_physics_ms/PLAY_TICKS);
    printf("  PARENT 명령서 생성:    %8.2f ms  (틱평균 %.2f ms)\n",
           m->play_cmd_ms, m->play_cmd_ms/PLAY_TICKS);
    printf("  CHILD  렌더링:         %8.2f ms  (틱평균 %.2f ms)\n",
           m->play_render_ms, m->play_render_ms/PLAY_TICKS);
    printf("  플레이 벽시계(병렬):   %8.2f ms\n", m->play_wall_ms);
    printf("===========================================================\n");
    printf("  총 소요 시간:          %8.2f ms\n", m->total_ms);
    printf("===========================================================\n");
    printf("[성능 지표]\n");
    printf("  Baseline Wall Time:    %8.2f ms\n", BASELINE_WALL_MS);
    printf("  Speedup:               %8.2fx\n", speedup);
    printf("  Efficiency:            %8.1f%%  (= Speedup / 2코어)\n", efficiency);
    printf("===========================================================\n");
    printf("[프로세스 생성/종료]\n");
    printf("  fork():                %8.3f ms\n", m->fork_ms);
    printf("  waitpid() 세팅:        %8.3f ms\n", m->waitpid_setting_ms);
    printf("  waitpid() 플레이:      %8.3f ms\n", m->waitpid_play_ms);
    printf("===========================================================\n");
    printf("[IPC pipe 오버헤드]\n");
    printf("  tmap: 공유메모리 직접 참조 (0 bytes)\n");
    printf("  세팅 지형완료신호(read):  %7.3f ms |       1 bytes\n",
           m->pipe_read_setting_ms);
    printf("  플레이 cmd신호(write):    %7.3f ms |  %4ld bytes\n",
           m->pipe_write_play_ms, m->pipe_bytes_play);
    printf("  플레이 렌더결과(read):    %7.3f ms |  %4ld bytes\n",
           m->pipe_read_result_ms,
           (long)(PLAY_TICKS * (long)sizeof(TickResult)));
    printf("  IPC 오버헤드 합계:        %7.3f ms\n",
           m->pipe_read_setting_ms + m->pipe_write_play_ms + m->pipe_read_result_ms);
    printf("===========================================================\n");
    printf("[CPU 성능 지표]\n");
    printf("  CPU user: %8.2f ms | sys: %8.2f ms\n",
           m->cpu_user_ms, m->cpu_sys_ms);
    printf("  CPU 활용률: %.1f%%\n", cpu_util);
    printf("===========================================================\n");
    printf("[세팅 단계 코어별]        [플레이 단계 코어별]\n");
    for(int i=0;i<m->num_cores && i<16;i++) {
        const char *slabel = (i==CORE_PARENT)?"(PARENT)":
                             (i==CORE_CHILD) ?"(CHILD) ":"       ";
        const char *plabel = (i==CORE_PARENT)?"(PARENT)":
                             (i==CORE_CHILD) ?"(CHILD) ":"       ";
        printf("  cpu%2d%s: %5.1f%%          cpu%2d%s: %5.1f%%\n",
               i, slabel, m->setup_core_util[i],
               i, plabel, m->play_core_util[i]);
    }
    printf("===========================================================\n");
    printf("[컨텍스트 스위칭]\n");
    printf("               voluntary    nonvoluntary\n");
    printf("  세팅 단계:   %6ld 회     %6ld 회\n",
           m->setup_vol_ctx, m->setup_nonvol_ctx);
    printf("  플레이 단계: %6ld 회     %6ld 회\n",
           m->play_vol_ctx, m->play_nonvol_ctx);
    printf("===========================================================\n");
    printf("[메모리 사용량]\n");
    printf("  세팅 완료 후 RSS:  %6ld KB | Peak: %6ld KB\n",
           m->mem_rss_setup_kb, m->mem_peak_setup_kb);
    printf("  플레이 완료 후 RSS:%6ld KB | Peak: %6ld KB\n",
           m->mem_rss_play_kb, m->mem_peak_play_kb);
    printf("===========================================================\n");
    printf("[결과 검증 — Baseline과 동일해야 함]\n");
    printf("  평균 조도:     %.6f  (기준: 0.397698) %s\n",
           m->lighting_result,
           fabsf(m->lighting_result-0.397698f)<1e-4f?"✓":"✗ RC 의심");
    printf("  AI 최고점수:   %.4f  (기준: 97.7063)  %s\n",
           m->ai_best_score,
           fabsf(m->ai_best_score-97.7063f)<1e-2f?"✓":"✗ RC 의심");
    printf("  파티클 에너지: %.6f  (기준: 25324.066406) %s\n",
           m->particle_final_energy,
           fabsf(m->particle_final_energy-25324.066406f)<1.0f?"✓":"✗ RC 의심");
    printf("  무결성 실패:   %d    (기준: 0) %s\n",
           m->integrity_fails,
           m->integrity_fails==0?"✓":"✗ 데이터 손상");
    printf("===========================================================\n");
}

/* ============================================================
 * 섹션 7: main
 * ============================================================ */
int main(void) {
    printf("=== 가 실험: PARENT1 + CHILD1 병렬 ===\n");
    printf("시드: 0x%08X | 맵: %dx%d | 청크: %d개\n\n",
           FIXED_SEED, MAP_WIDTH, MAP_HEIGHT, TOTAL_CHUNKS);

    PerfMetrics m = {0};
    double t0 = now_ms();

    /* ── 공유메모리 생성 (tmap: ~4MB + ChunkDesc 포인터 제외) ── */
    /* ChunkDesc.height_data 포인터는 프로세스 고유 힙에 있으므로
       공유 불가 → height_data는 각자 할당, tmap->data만 공유 */
    size_t shm_size = sizeof(TerrainMap);
    int shm_id = shmget(IPC_PRIVATE, shm_size, IPC_CREAT | 0666);
    if (shm_id < 0) { perror("shmget"); return 1; }
    TerrainMap *tmap = (TerrainMap*)shmat(shm_id, NULL, 0);
    if (tmap == (void*)-1) { perror("shmat"); return 1; }
    memset(tmap, 0, shm_size);

    /* ── chunk height_data는 CHILD만 사용 (fork 전 할당 → CHILD가 CoW 후 해제) ── */
    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        int cx = i % CHUNKS_PER_ROW, cy = i / CHUNKS_PER_ROW;
        tmap->chunks[i].chunk_id = i;
        tmap->chunks[i].chunk_x  = cx;
        tmap->chunks[i].chunk_y  = cy;
        tmap->chunks[i].pixel_x_start = cx * CHUNK_SIZE;
        tmap->chunks[i].pixel_y_start = cy * CHUNK_SIZE;
        tmap->chunks[i].height_data = malloc(sizeof(float)*CHUNK_SIZE*CHUNK_SIZE);
        if (!tmap->chunks[i].height_data) { perror("malloc"); return 1; }
    }

    /* ── pipe 생성 (fork 전) ── */
    int terrain_done_pipe[2], cmd_ready_pipe[2], render_done_pipe[2];
    if (pipe(terrain_done_pipe) < 0) { perror("pipe terrain_done"); return 1; }
    if (pipe(cmd_ready_pipe)    < 0) { perror("pipe cmd_ready");    return 1; }
    if (pipe(render_done_pipe)  < 0) { perror("pipe render_done");  return 1; }

    /* ── cmds 공유메모리 (PARENT가 매 틱 기록 → CHILD가 렌더링에 사용) ── */
    int cmds_shm_id = shmget(IPC_PRIVATE, sizeof(DrawCall) * DRAWCALL_COUNT, IPC_CREAT | 0666);
    if (cmds_shm_id < 0) { perror("shmget cmds"); return 1; }
    DrawCall *cmds = (DrawCall*)shmat(cmds_shm_id, NULL, 0);
    if (cmds == (void*)-1) { perror("shmat cmds"); return 1; }
    memset(cmds, 0, sizeof(DrawCall) * DRAWCALL_COUNT);

    /* ── 코어 고정: PARENT → cpu0 ── */
    pin_to_core(CORE_PARENT);
    printf("[PIN] PARENT → cpu%d\n", CORE_PARENT);

    /* ── 측정 시작 스냅샷 ── */
    CpuSnapshot snap_setup_before, snap_setup_after;
    CpuSnapshot snap_play_before,  snap_play_after;
    CtxSwitch ctx_setup_before, ctx_setup_after;
    CtxSwitch ctx_play_before,  ctx_play_after;
    cpu_snapshot(&snap_setup_before);
    read_ctx_switches(&ctx_setup_before);

    /* ── fork() ── */
    double t_fork = now_ms();
    pid_t child_pid = fork();
    m.fork_ms = now_ms() - t_fork;

    if (child_pid < 0) { perror("fork"); return 1; }

    /* ============================================================
     * CHILD 프로세스: 지형맵 생성 → tmap->data(공유메모리) 기록 후 종료
     * ============================================================ */
    if (child_pid == 0) {
        pin_to_core(CORE_CHILD);

        /* 사용하지 않는 pipe 끝 닫기 */
        close(terrain_done_pipe[0]);
        close(cmd_ready_pipe[1]);
        close(render_done_pipe[0]);

        double child_t0 = now_ms();

        /* [세팅] 지형맵 생성 */
        init_noise_params(&tmap->params);

        printf("\n[CHILD] 단계1: 노이즈 연산 (%d청크)...\n", TOTAL_CHUNKS);
        double t = now_ms();
        for (int ci = 0; ci < TOTAL_CHUNKS; ci++) {
            noise_compute_chunk(&tmap->chunks[ci], &tmap->params);
            if ((ci+1)%64==0)
                printf("[CHILD] %d/%d (%.0f%%)\n",
                       ci+1, TOTAL_CHUNKS, (ci+1)*100.0/TOTAL_CHUNKS);
        }
        printf("[CHILD] 노이즈 완료: %.2f ms\n", now_ms()-t);

        t = now_ms();
        printf("[CHILD] 단계2: 청크 병합...\n");
        terrain_merge(tmap);
        printf("[CHILD] 병합 완료: %.2f ms\n", now_ms()-t);

        t = now_ms();
        printf("[CHILD] 단계3: 경계 블러...\n");
        terrain_blur(tmap);
        printf("[CHILD] 블러 완료: %.2f ms\n", now_ms()-t);

        t = now_ms();
        printf("[CHILD] 단계4: 수력 침식...\n");
        terrain_erosion(tmap);
        printf("[CHILD] 침식 완료: %.2f ms\n", now_ms()-t);

        t = now_ms();
        printf("[CHILD] 단계5: 히스토그램 + 무결성...\n");
        terrain_histogram(tmap);
        tmap->integrity_fails    = terrain_integrity(tmap);
        tmap->terrain_elapsed_ms = now_ms() - child_t0;
        printf("[CHILD] 지형 생성 완료: %.2f ms | 무결성 실패: %d\n\n",
               tmap->terrain_elapsed_ms, tmap->integrity_fails);

        for (int i = 0; i < TOTAL_CHUNKS; i++)
            free(tmap->chunks[i].height_data);

        /* 세팅 완료 신호 → PARENT */
        { char sig = 1; write_all(terrain_done_pipe[1], &sig, 1); }
        close(terrain_done_pipe[1]);

        /* [플레이] CHILD 렌더 루프: pipe 신호 대기 → tick_render() → 결과 반환 */
        printf("[CHILD] 플레이 렌더 루프 시작 (%d틱)...\n", PLAY_TICKS);
        for (int tick = 0; tick < PLAY_TICKS; tick++) {
            char go;
            read_all(cmd_ready_pipe[0], &go, 1);

            double t_render = now_ms();
            double pixel_sum = tick_render(cmds, tmap, tick);
            double render_ms = now_ms() - t_render;

            TickResult tr = {pixel_sum, render_ms};
            write_all(render_done_pipe[1], &tr, sizeof(tr));
        }
        printf("[CHILD] 플레이 렌더 루프 완료\n");

        close(cmd_ready_pipe[0]);
        close(render_done_pipe[1]);
        shmdt(tmap);
        shmdt(cmds);
        exit(0);
    }

    /* ============================================================
     * PARENT 프로세스: 조도+AI+파티클 → pipe 신호 수신 → 플레이 병렬 실행
     * ============================================================ */

    /* 사용하지 않는 pipe 끝 닫기 */
    close(terrain_done_pipe[1]);
    close(cmd_ready_pipe[0]);
    close(render_done_pipe[1]);

    /* [세팅] 조도 + AI + 파티클 (CHILD 지형 생성과 병렬) */
    double t_lighting = now_ms();
    m.lighting_result = lighting_simulation(FIXED_SEED);
    m.lighting_ms     = now_ms() - t_lighting;
    printf("[PARENT] 조도 완료: %.2f ms\n\n", m.lighting_ms);

    double t_ai = now_ms();
    m.ai_best_score = monster_ai(FIXED_SEED);
    m.ai_ms         = now_ms() - t_ai;
    printf("[PARENT] AI 완료: %.2f ms\n\n", m.ai_ms);

    double t_particle = now_ms();
    m.particle_final_energy = particle_simulation(FIXED_SEED);
    m.particle_ms           = now_ms() - t_particle;
    printf("[PARENT] 파티클 완료: %.2f ms\n\n", m.particle_ms);

    /* CHILD 지형 생성 완료 신호 대기 (pipe) — 이후 tmap->data 안전하게 참조 가능 */
    double t_wait_s = now_ms();
    { char terrain_sig;
      double t_pr = now_ms();
      read_all(terrain_done_pipe[0], &terrain_sig, 1);
      m.pipe_read_setting_ms = now_ms() - t_pr; }
    close(terrain_done_pipe[0]);
    m.waitpid_setting_ms = now_ms() - t_wait_s;
    m.pipe_bytes_setting = 1;
    m.terrain_ms         = tmap->terrain_elapsed_ms;
    m.integrity_fails    = tmap->integrity_fails;

    m.setting_wall_ms = now_ms() - t0;

    /* 세팅 단계 측정 완료 */
    cpu_snapshot(&snap_setup_after);
    read_ctx_switches(&ctx_setup_after);
    cpu_calc_util(&snap_setup_before, &snap_setup_after,
                  m.setup_core_util, &m.num_cores);
    m.setup_vol_ctx    = ctx_setup_after.voluntary    - ctx_setup_before.voluntary;
    m.setup_nonvol_ctx = ctx_setup_after.nonvoluntary - ctx_setup_before.nonvoluntary;
    read_memory_usage(&m.mem_rss_setup_kb, &m.mem_peak_setup_kb);

    printf("\n[세팅 완료] 벽시계: %.2f ms\n\n", m.setting_wall_ms);

    /* [플레이] PARENT(물리+명령서) || CHILD(렌더) 병렬 실행 */
    cpu_snapshot(&snap_play_before);
    read_ctx_switches(&ctx_play_before);

    uint32_t rng = FIXED_SEED ^ 0xCAFEF00D;
    double t_physics_total  = 0.0, t_cmd_total = 0.0, t_render_total = 0.0;
    double t_pipe_write_play = 0.0, t_pipe_read_result = 0.0;
    double pixel_grand_total = 0.0;
    double t_play = now_ms();

    printf("[PLAY] 시작: %d틱 | PARENT(물리+명령서) || CHILD(렌더)\n", PLAY_TICKS);

    for (int tick = 0; tick < PLAY_TICKS; tick++) {
        double tp = now_ms();
        int hits = tick_physics_ai(tmap, &rng, tick);
        t_physics_total += now_ms() - tp;

        double tc = now_ms();
        tick_build_commands(tmap, cmds, hits, &rng);
        t_cmd_total += now_ms() - tc;

        /* cmds는 공유메모리에 이미 기록됨 → CHILD에 준비 신호 */
        { char go = 1;
          double tw = now_ms();
          write_all(cmd_ready_pipe[1], &go, 1);
          t_pipe_write_play += now_ms() - tw; }

        /* CHILD 렌더 결과 수신 */
        TickResult tr;
        { double tr_t = now_ms();
          read_all(render_done_pipe[0], &tr, sizeof(tr));
          t_pipe_read_result += now_ms() - tr_t; }

        pixel_grand_total += tr.pixel_sum;
        t_render_total    += tr.render_ms;

        if ((tick+1) % (PLAY_TICKS/10) == 0)
            printf("[PLAY] 틱 %3d/%d | 물리: %.1fms | 명령서: %.1fms | 렌더(CHILD): %.1fms\n",
                   tick+1, PLAY_TICKS, t_physics_total, t_cmd_total, t_render_total);
    }

    close(cmd_ready_pipe[1]);
    close(render_done_pipe[0]);

    m.play_wall_ms        = now_ms() - t_play;
    m.play_physics_ms     = t_physics_total;
    m.play_cmd_ms         = t_cmd_total;
    m.play_render_ms      = t_render_total;
    m.pixel_grand         = pixel_grand_total;
    m.pipe_write_play_ms  = t_pipe_write_play;
    m.pipe_read_result_ms = t_pipe_read_result;
    m.pipe_bytes_play     = PLAY_TICKS;  /* cmd 준비 신호: 1 byte × 틱 */

    printf("[PLAY] 완료: %.2f ms\n\n", m.play_wall_ms);

    /* CHILD 최종 종료 대기 */
    { int status;
      double t_wait_play = now_ms();
      waitpid(child_pid, &status, 0);
      m.waitpid_play_ms = now_ms() - t_wait_play; }

    m.total_ms = now_ms() - t0;

    /* 플레이 단계 측정 완료 */
    cpu_snapshot(&snap_play_after);
    read_ctx_switches(&ctx_play_after);
    cpu_calc_util(&snap_play_before, &snap_play_after,
                  m.play_core_util, &m.num_cores);
    m.play_vol_ctx    = ctx_play_after.voluntary    - ctx_play_before.voluntary;
    m.play_nonvol_ctx = ctx_play_after.nonvoluntary - ctx_play_before.nonvoluntary;
    read_memory_usage(&m.mem_rss_play_kb, &m.mem_peak_play_kb);

    /* CPU 전체 */
    struct rusage ru_s, ru_c;
    getrusage(RUSAGE_SELF,     &ru_s);
    getrusage(RUSAGE_CHILDREN, &ru_c);
    m.cpu_user_ms = (ru_s.ru_utime.tv_sec  + ru_c.ru_utime.tv_sec)  * 1e3
                  + (ru_s.ru_utime.tv_usec + ru_c.ru_utime.tv_usec) / 1e3;
    m.cpu_sys_ms  = (ru_s.ru_stime.tv_sec  + ru_c.ru_stime.tv_sec)  * 1e3
                  + (ru_s.ru_stime.tv_usec + ru_c.ru_stime.tv_usec) / 1e3;

    /* 공유메모리 정리 — tmap/cmds 접근은 shmdt 전에 완료 */
    print_ascii_map(tmap);
    shmdt(tmap);
    shmctl(shm_id, IPC_RMID, NULL);
    shmdt(cmds);
    shmctl(cmds_shm_id, IPC_RMID, NULL);
    print_metrics(&m);

    return 0;
}
