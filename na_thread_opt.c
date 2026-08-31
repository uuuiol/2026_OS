#define _GNU_SOURCE
/*
 * ============================================================
 *  게임 월드 병렬화 실험 코드 — 나 실험 최적화 버전 — pthread_join 위치 수정
 *  (AI·파티클을 pthread_join 이전으로 이동하여 지형 Thread와 겹치게 함)
 * ============================================================
 *
 *  공통 기준:
 *    Seed, Map 크기, Chunk 크기, 반복 횟수, 계산식은 Baseline과 동일하게 유지한다.
 *    terrain_blur(), terrain_erosion(), terrain_histogram(), terrain_integrity()는 Baseline 계산 의미를 유지한다.
 *
 *  빌드:
 *    gcc -O2 -o na_thread1_play_fixed na_thread1_play_fixed.c -lm -lpthread
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/resource.h>   /* getrusage — CPU user+sys 시간 */

/* ============================================================
 * 섹션 0: 상수
 * ============================================================ */

#define FIXED_SEED      0xDEAD4096U

#define MAP_WIDTH       1024
#define MAP_HEIGHT      1024
#define CHUNK_SIZE      64
#define CHUNKS_PER_ROW  (MAP_WIDTH  / CHUNK_SIZE)
#define CHUNKS_PER_COL  (MAP_HEIGHT / CHUNK_SIZE)
#define TOTAL_CHUNKS    (CHUNKS_PER_ROW * CHUNKS_PER_COL)

/* 병렬화 시 참조 */
#define NUM_THREADS          4
#define CHUNKS_PER_THREAD    (TOTAL_CHUNKS / NUM_THREADS)
#define ROWS_PER_THREAD      (MAP_HEIGHT   / NUM_THREADS)

/* [THREAD] 노이즈 */
#define OCTAVE_COUNT    6
#define MC_SAMPLES      160
#define BLUR_RADIUS     2

/* [THREAD] 침식 */
#define EROSION_DROPS   ((long)MAP_WIDTH * MAP_HEIGHT * 16)
#define EROSION_STEPS   256

/* 검사 */
#define MAX_CLIFF_DIFF  0.6f
#define HEIGHT_MIN      0.0f
#define HEIGHT_MAX      1.0f
#define HISTOGRAM_BINS  256

/* [PARENT] 조도 계산 파라미터 */
#define LIGHT_SAMPLES   200000000L

/* [CHILD1] 몬스터 AI 파라미터 */
#define AI_MONSTERS     20
#define AI_WEIGHTS      16
#define AI_ANNEAL_STEPS 325
#define AI_COMBAT_SIM   500
#define AI_COMBAT_TICKS 300

/* [CHILD2] 파티클 파라미터 */
#define PARTICLE_TYPES  4
#define PARTICLE_COUNT  400
#define PARTICLE_STEPS  2250

/*
 * [PLAY] 틱 기반 GPU 커맨드 버퍼 파이프라인 파라미터
 *
 *   PLAY_TICKS        : 총 틱 수 (연산량의 주 조절 변수)
 *   PHYSICS_OBJECTS   : 틱당 물리 시뮬레이션 오브젝트 수
 *   AI_SIGHT_RAYS     : 틱당 시야 레이캐스트 수 (오브젝트당)
 *   RENDER_PASSES     : 틱당 렌더링 패스 수 (GPU 연산 가정)
 *   DRAWCALL_COUNT    : 틱당 드로우콜 명령서 수
 *
 *   틱 흐름:
 *     [PARENT] 물리/AI 연산 → 명령서 생성 → [THREAD] GPU 렌더링
 *   병렬화 시: 이전 틱 GPU 렌더링과 현재 틱 CPU 연산이 겹침
 */
#define PLAY_TICKS          500
#define PHYSICS_OBJECTS     5000
#define AI_SIGHT_RAYS       128
#define RENDER_PASSES       8
#define DRAWCALL_COUNT      2048
#define RAYMARCH_STEPS      6000
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
} TerrainMap;

typedef struct {
    float x, y, z, vx, vy, vz, fx, fy, fz;
    float mass, charge, lifetime;
} Particle;

/* 드로우콜 명령서 (PARENT → THREAD) */
typedef struct {
    int   object_id;
    float pos_x, pos_y, pos_z;
    float normal_x, normal_y, normal_z;
    float light_intensity;
    int   pass_mask;        /* 렌더링 패스 비트마스크 */
} DrawCall;

typedef struct {
    /* 세팅 단계 */
    double thread_noise_ms, thread_blur_ms;
    double thread_erosion_ms, thread_merge_ms;
    double thread_integrity_ms, thread_total_ms;
    double parent_lighting_ms;
    double child1_ai_ms;
    double child2_particle_ms;
    /* 플레이 단계 */
    double play_roundrobin_ms;  /* [PLAY] 총 틱 루프 시간 */
    int    play_ticks;          /* 실행한 틱 수 */
    double play_physics_ms;     /* PARENT 물리/AI 누적 시간 */
    double play_cmd_ms;         /* PARENT 명령서 생성 누적 시간 */
    double play_render_ms;      /* THREAD GPU 렌더링 누적 시간 */
    double play_thread_create_ms; /* 플레이 단계 render pthread_create 오버헤드 */
    /* 전체 */
    double total_ms;
    /* CPU 성능 지표 */
    double cpu_user_ms;
    double cpu_sys_ms;
    /* IPC (베이스라인은 0, 병렬화 시 채워짐) */
    int    ipc_pipe_write;
    int    ipc_pipe_read;
    int    ipc_bytes;
    double ipc_overhead_ms;
    /* 결과 검증 */
    int    integrity_fails;
    float  lighting_result;
    float  ai_best_score;
    float  particle_final_energy;
} PerfMetrics;

/* ============================================================
 * 섹션 2: 유틸리티
 * ============================================================ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static uint32_t lcg(uint32_t *s) { *s = (*s) * 1664525u + 1013904223u; return *s; }
static float lcg_f(uint32_t *s) { return (float)(lcg(s) & 0xFFFFFF) / (float)0x1000000; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

static int role_cpu(int role_index) {
    int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (cores <= 1) return 0;
    if (role_index < cores) return role_index;
    return 1 + ((role_index - 1) % (cores - 1));
}

static void pin_current_thread_to_cpu(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}

/* ============================================================
 * 섹션 3: [THREAD] 지형 생성
 * ============================================================ */

static float perlin_octave(float x, float y, float freq, float amp, uint32_t seed) {
    float fx = x * freq, fy = y * freq;
    int ix = (int)floorf(fx), iy = (int)floorf(fy);
    float tx = fx - ix, ty = fy - iy;
    float ux = tx*tx*tx*(tx*(tx*6-15)+10);
    float uy = ty*ty*ty*(ty*(ty*6-15)+10);
    uint32_t s00 = seed ^ (uint32_t)(ix*1619 + iy*31337);
    uint32_t s10 = seed ^ (uint32_t)((ix+1)*1619 + iy*31337);
    uint32_t s01 = seed ^ (uint32_t)(ix*1619 + (iy+1)*31337);
    uint32_t s11 = seed ^ (uint32_t)((ix+1)*1619 + (iy+1)*31337);
    float g00 = sinf((float)s00*1e-5f)*cosf((float)(s00>>8)*1e-5f);
    float g10 = sinf((float)s10*1e-5f)*cosf((float)(s10>>8)*1e-5f);
    float g01 = sinf((float)s01*1e-5f)*cosf((float)(s01>>8)*1e-5f);
    float g11 = sinf((float)s11*1e-5f)*cosf((float)(s11>>8)*1e-5f);
    float v0 = g00 + ux*(g10-g00), v1 = g01 + ux*(g11-g01);
    return amp * (v0 + uy*(v1-v0));
}

static float octave_stack(float px, float py, const NoiseParams *p) {
    float h = 0;
    for (int oct = 0; oct < OCTAVE_COUNT; oct++)
        h += perlin_octave(px, py, p->frequency[oct], p->amplitude[oct],
                           p->seed ^ (uint32_t)(oct * 0xDEAD));
    return h;
}

static float monte_carlo(float px, float py, const NoiseParams *p) {
    uint32_t rng = p->seed ^ (uint32_t)(px*7919 + py*6271);
    float sum = 0, wsum = 0;
    for (int s = 0; s < MC_SAMPLES; s++) {
        float jx = px + (lcg_f(&rng)-0.5f)*0.1f;
        float jy = py + (lcg_f(&rng)-0.5f)*0.1f;
        float h = octave_stack(jx, jy, p);
        float dx = jx-px, dy = jy-py;
        float w = expf(-(dx*dx+dy*dy)*8);
        sum += h*w; wsum += w;
    }
    return wsum > 0 ? sum/wsum : 0;
}

static void chunk_blur(float *data, int w, int h) {
    static const float K[5][5] = {
        {0.00390625f,0.015625f,0.0234375f,0.015625f,0.00390625f},
        {0.015625f,  0.0625f,  0.09375f,  0.0625f,  0.015625f  },
        {0.0234375f, 0.09375f, 0.140625f, 0.09375f, 0.0234375f },
        {0.015625f,  0.0625f,  0.09375f,  0.0625f,  0.015625f  },
        {0.00390625f,0.015625f,0.0234375f,0.015625f,0.00390625f},
    };
    float *tmp = malloc(sizeof(float)*(size_t)(w*h));
    if (!tmp) return;
    memcpy(tmp, data, sizeof(float)*(size_t)(w*h));
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float acc = 0, ws = 0;
        for (int ky = -2; ky <= 2; ky++) for (int kx = -2; kx <= 2; kx++) {
            int nx = x+kx, ny = y+ky;
            if (nx<0||nx>=w||ny<0||ny>=h) continue;
            float kv = K[ky+2][kx+2];
            acc += tmp[ny*w+nx]*kv; ws += kv;
        }
        data[y*w+x] = ws > 0 ? acc/ws : tmp[y*w+x];
    }
    free(tmp);
}

static void noise_compute_chunk(ChunkDesc *c, const NoiseParams *p) {
    for (int ly = 0; ly < CHUNK_SIZE; ly++) for (int lx = 0; lx < CHUNK_SIZE; lx++) {
        float px = (float)(c->pixel_x_start+lx)/MAP_WIDTH;
        float py = (float)(c->pixel_y_start+ly)/MAP_HEIGHT;
        float h = monte_carlo(px, py, p);
        c->height_data[ly*CHUNK_SIZE+lx] = clampf((h+1)*0.5f, HEIGHT_MIN, HEIGHT_MAX);
    }
    chunk_blur(c->height_data, CHUNK_SIZE, CHUNK_SIZE);
}

static void terrain_blur(TerrainMap *tmap) {
    float (*map)[MAP_WIDTH] = tmap->data;
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

static void terrain_erosion(TerrainMap *tmap) {
    float (*map)[MAP_WIDTH] = tmap->data;
    const float er=0.004f, dr=0.002f;
    uint32_t rng = tmap->params.seed ^ 0xBAADF00D;
    for (long d = 0; d < EROSION_DROPS; d++) {
        int fx=(int)(((rng=rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_WIDTH-2))+1;
        int fy=(int)(((rng=rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_HEIGHT-2))+1;
        for (int step = 0; step < EROSION_STEPS; step++) {
            if (fx<1||fx>=MAP_WIDTH-1||fy<1||fy>=MAP_HEIGHT-1) break;
            float hc=map[fy][fx], hm=hc; int dx=0, dy=0;
            if (map[fy][fx+1]<hm){hm=map[fy][fx+1];dx= 1;dy= 0;}
            if (map[fy][fx-1]<hm){hm=map[fy][fx-1];dx=-1;dy= 0;}
            if (map[fy+1][fx]<hm){hm=map[fy+1][fx];dx= 0;dy= 1;}
            if (map[fy-1][fx]<hm){hm=map[fy-1][fx];dx= 0;dy=-1;}
            if (!dx&&!dy) break;
            float diff=hc-hm;
            map[fy][fx]-=er*diff; map[fy+dy][fx+dx]+=dr*diff;
            fx+=dx; fy+=dy;
        }
    }
    for (int y=0;y<MAP_HEIGHT;y++) for (int x=0;x<MAP_WIDTH;x++)
        map[y][x]=clampf(map[y][x],HEIGHT_MIN,HEIGHT_MAX);
}

static void terrain_merge(TerrainMap *tmap) {
    for (int ci = 0; ci < TOTAL_CHUNKS; ci++) {
        ChunkDesc *c = &tmap->chunks[ci];
        for (int ly = 0; ly < CHUNK_SIZE; ly++) {
            int gy = c->pixel_y_start+ly;
            memcpy(&tmap->data[gy][c->pixel_x_start],
                   &c->height_data[ly*CHUNK_SIZE],
                   sizeof(float)*CHUNK_SIZE);
        }
    }
}

static void terrain_histogram(TerrainMap *tmap) {
    float (*map)[MAP_WIDTH] = tmap->data;
    uint32_t hist[HISTOGRAM_BINS] = {0};
    for (int y=0;y<MAP_HEIGHT;y++) for (int x=0;x<MAP_WIDTH;x++) {
        int b=(int)(map[y][x]*(HISTOGRAM_BINS-1));
        b=b<0?0:b>=HISTOGRAM_BINS?HISTOGRAM_BINS-1:b; hist[b]++;
    }
    float cdf[HISTOGRAM_BINS]; uint32_t cum=0,cmin=0; int found=0;
    for (int b=0;b<HISTOGRAM_BINS;b++) {
        cum+=hist[b]; cdf[b]=(float)cum;
        if (!found&&hist[b]>0){cmin=cum;found=1;}
    }
    uint32_t tot=MAP_WIDTH*MAP_HEIGHT;
    for (int b=0;b<HISTOGRAM_BINS;b++) {
        cdf[b]=(cdf[b]-(float)cmin)/(float)(tot-cmin);
        cdf[b]=clampf(cdf[b],0,1);
    }
    for (int y=0;y<MAP_HEIGHT;y++) for (int x=0;x<MAP_WIDTH;x++) {
        int b=(int)(map[y][x]*(HISTOGRAM_BINS-1));
        b=b<0?0:b>=HISTOGRAM_BINS?HISTOGRAM_BINS-1:b; map[y][x]=cdf[b];
    }
}

static int terrain_integrity(const TerrainMap *tmap) {
    int fail = 0;
    for (int y=0;y<MAP_HEIGHT;y++) for (int x=0;x<MAP_WIDTH;x++) {
        float h=tmap->data[y][x];
        if (h<HEIGHT_MIN-1e-4f||h>HEIGHT_MAX+1e-4f){fail++;continue;}
        if (x+1<MAP_WIDTH&&fabsf(h-tmap->data[y][x+1])>MAX_CLIFF_DIFF) fail++;
        if (y+1<MAP_HEIGHT&&fabsf(h-tmap->data[y+1][x])>MAX_CLIFF_DIFF) fail++;
    }
    return fail;
}

static void init_noise_params(NoiseParams *p) {
    p->seed=FIXED_SEED; p->persistence=0.5f; p->lacunarity=2.0f;
    uint32_t rng=FIXED_SEED;
    p->mountain_weight=0.3f+lcg_f(&rng)*0.4f;
    p->roughness      =0.4f+lcg_f(&rng)*0.4f;
    float freq=1,amp=1,asum=0;
    for (int i=0;i<OCTAVE_COUNT;i++) {
        p->frequency[i]=freq*p->roughness; p->amplitude[i]=amp;
        asum+=amp; freq*=p->lacunarity; amp*=p->persistence;
    }
    for (int i=0;i<OCTAVE_COUNT;i++) p->amplitude[i]/=asum;
    printf("[THREAD] Seed: 0x%08X | Mountain: %.2f | Roughness: %.2f\n",
           p->seed, p->mountain_weight, p->roughness);
}

/* ============================================================
 * 섹션 4: [PARENT] 조도 계산
 * ============================================================ */
static float parent_lighting_simulation(uint32_t seed) {
    printf("[PARENT] 조도 계산 시작 (%ldM 샘플)...\n", LIGHT_SAMPLES/1000000L);
    uint32_t rng = seed ^ 0xABCD1234;
    double total = 0.0;
    float sun_az  = lcg_f(&rng)*2.0f*3.14159f;
    float sun_alt = 0.2f+lcg_f(&rng)*0.6f;
    float sun_x   = cosf(sun_alt)*cosf(sun_az);
    float sun_y   = sinf(sun_alt);
    float sun_z   = cosf(sun_alt)*sinf(sun_az);
    for (long i = 0; i < LIGHT_SAMPLES; i++) {
        float sx=lcg_f(&rng), sy=lcg_f(&rng);
        float u=lcg_f(&rng), v=lcg_f(&rng);
        float theta=acosf(sqrtf(u));
        float phi=2.0f*3.14159f*v;
        float rx=sinf(theta)*cosf(phi);
        float ry=cosf(theta);
        float rz=sinf(theta)*sinf(phi);
        float direct=rx*sun_x+ry*sun_y+rz*sun_z;
        if (direct<0) direct=0;
        float scatter=expf(-sx*sx*2.0f-sy*sy*2.0f)*0.3f;
        total += (double)(direct+scatter);
    }
    float result = (float)(total/LIGHT_SAMPLES);
    printf("[PARENT] 조도 계산 완료. 평균 조도: %.6f\n", result);
    return result;
}

/* ============================================================
 * 섹션 5: [CHILD1] 몬스터 AI 행동트리 최적화
 * ============================================================ */
static float child1_monster_ai(uint32_t seed) {
    printf("[CHILD1] 몬스터 AI 최적화 시작...\n");
    printf("         %d종 × %d스텝 × %d전투 × %d틱\n",
           AI_MONSTERS, AI_ANNEAL_STEPS, AI_COMBAT_SIM, AI_COMBAT_TICKS);
    uint32_t rng = seed ^ 0xDEADBEEF;
    float global_best = 0.0f;
    for (int m = 0; m < AI_MONSTERS; m++) {
        float w[AI_WEIGHTS], best_w[AI_WEIGHTS];
        for (int i=0;i<AI_WEIGHTS;i++) w[i]=best_w[i]=lcg_f(&rng)*2-1;
        float best_score=-1e9f;
        float T=1.0f;
        const float cool=1.0f-(1.0f-1e-4f)/AI_ANNEAL_STEPS;
        for (int step=0;step<AI_ANNEAL_STEPS;step++) {
            float cand[AI_WEIGHTS];
            for (int i=0;i<AI_WEIGHTS;i++)
                cand[i]=w[i]+(lcg_f(&rng)-0.5f)*T*0.5f;
            double score_sum=0;
            for (int sim=0;sim<AI_COMBAT_SIM;sim++) {
                float mhp=100,php=100;
                float matk=fabsf(cand[0])*10+5;
                float aggr=clampf(cand[1]*0.5f+0.5f,0,1);
                float evad=clampf(cand[2]*0.5f+0.5f,0,1);
                for (int tick=0;tick<AI_COMBAT_TICKS;tick++) {
                    if (mhp<=0||php<=0) break;
                    float hr=mhp/100.0f;
                    float atk_s=cand[3]*hr+cand[4]*aggr;
                    float ret_s=cand[5]*(1-hr)+cand[6]*evad;
                    if (atk_s>ret_s) {
                        float dmg=matk*(1+cand[7]*0.2f);
                        for (int k=0;k<AI_WEIGHTS;k++) dmg+=cand[k]*cand[k]*0.01f;
                        php-=dmg*0.016f;
                        mhp-=8*(1-evad*0.3f)*0.016f;
                    } else {
                        float ev=evad*(1+sinf((float)tick*0.1f)*0.2f);
                        mhp-=8*(1-ev)*0.016f;
                    }
                    mhp=clampf(mhp,0,100); php=clampf(php,0,100);
                }
                score_sum+=(100-php)*0.6f+mhp*0.4f;
            }
            float score=(float)(score_sum/AI_COMBAT_SIM);
            float delta=score-best_score;
            if (delta>0||lcg_f(&rng)<expf(delta/T)) {
                memcpy(w,cand,sizeof(w));
                if (score>best_score){best_score=score;memcpy(best_w,w,sizeof(w));}
            }
            T*=cool;
        }
        if (best_score>global_best) global_best=best_score;
        if ((m+1)%5==0)
            printf("[CHILD1] %d/%d 몬스터 완료 (최고: %.2f)\n",
                   m+1, AI_MONSTERS, global_best);
    }
    printf("[CHILD1] AI 최적화 완료. 최고 점수: %.4f\n", global_best);
    return global_best;
}

/* ============================================================
 * 섹션 6: [CHILD2] 파티클 시뮬레이션
 * ============================================================ */
typedef enum { EXPLOSION=0, SPLASH, LAVA, SMOKE } EffectType;

static float simulate_effect(EffectType type, uint32_t seed) {
    const char *names[]={"폭발","물보라","용암","연기"};
    float gravity,viscosity,init_speed,eps,sigma,lt_max;
    switch(type){
        case EXPLOSION: gravity=9.8f;viscosity=0.01f;init_speed=15;eps=0.5f;sigma=0.3f;lt_max=2;break;
        case SPLASH:    gravity=9.8f;viscosity=0.1f; init_speed=5; eps=1.0f;sigma=0.5f;lt_max=4;break;
        case LAVA:      gravity=9.8f;viscosity=2.0f; init_speed=2; eps=2.0f;sigma=0.8f;lt_max=8;break;
        case SMOKE:     gravity=-1.f;viscosity=0.05f;init_speed=1; eps=0.1f;sigma=1.0f;lt_max=12;break;
        default:        gravity=9.8f;viscosity=0.1f; init_speed=5; eps=1.0f;sigma=0.5f;lt_max=4;break;
    }
    printf("[CHILD2] %s 시뮬레이션 (%d개×%d스텝)...\n",
           names[type], PARTICLE_COUNT, PARTICLE_STEPS);
    Particle *p = malloc(sizeof(Particle)*PARTICLE_COUNT);
    if (!p) return 0;
    uint32_t rng = seed ^ (uint32_t)(type*0x1234);
    for (int i=0;i<PARTICLE_COUNT;i++) {
        p[i].x=(lcg_f(&rng)-0.5f)*2; p[i].y=(lcg_f(&rng)-0.5f)*2; p[i].z=(lcg_f(&rng)-0.5f)*2;
        float spd=init_speed*(0.5f+lcg_f(&rng)*0.5f);
        p[i].vx=(lcg_f(&rng)-0.5f)*spd;
        p[i].vy=(lcg_f(&rng)-0.5f)*spd;
        p[i].vz=(lcg_f(&rng)-0.5f)*spd;
        p[i].fx=p[i].fy=p[i].fz=0;
        p[i].mass=0.5f+lcg_f(&rng)*0.5f;
        p[i].charge=lcg_f(&rng)*2-1;
        p[i].lifetime=lt_max*(0.5f+lcg_f(&rng)*0.5f);
    }
    const float DT=0.001f;
    double total_ke=0;
    for (int step=0;step<PARTICLE_STEPS;step++) {
        for (int i=0;i<PARTICLE_COUNT;i++) p[i].fx=p[i].fy=p[i].fz=0;
        for (int i=0;i<PARTICLE_COUNT;i++) {
            if (p[i].lifetime<=0) continue;
            for (int j=i+1;j<PARTICLE_COUNT;j++) {
                if (p[j].lifetime<=0) continue;
                float dx=p[j].x-p[i].x, dy=p[j].y-p[i].y, dz=p[j].z-p[i].z;
                float r2=dx*dx+dy*dy+dz*dz;
                if (r2<0.01f) continue;
                float r=sqrtf(r2);
                float sr=sigma/r, sr6=sr*sr*sr*sr*sr*sr, sr12=sr6*sr6;
                float fmag=24*eps*(2*sr12-sr6)/r2;
                if (fmag> 1e5f) fmag= 1e5f;
                if (fmag<-1e5f) fmag=-1e5f;
                p[i].fx-=fmag*dx; p[i].fy-=fmag*dy; p[i].fz-=fmag*dz;
                p[j].fx+=fmag*dx; p[j].fy+=fmag*dy; p[j].fz+=fmag*dz;
            }
        }
        for (int i=0;i<PARTICLE_COUNT;i++) {
            if (p[i].lifetime<=0) continue;
            p[i].fy+=gravity*p[i].mass;
            p[i].fx-=viscosity*p[i].vx;
            p[i].fy-=viscosity*p[i].vy;
            p[i].fz-=viscosity*p[i].vz;
            if (type==SMOKE) {
                p[i].fx+=0.5f*sinf(p[i].y*3+(float)step*DT*2);
                p[i].fz+=0.5f*cosf(p[i].x*3+(float)step*DT*2);
            }
            p[i].vx+=p[i].fx/p[i].mass*DT;
            p[i].vy+=p[i].fy/p[i].mass*DT;
            p[i].vz+=p[i].fz/p[i].mass*DT;
            if (p[i].vx> 1e4f) p[i].vx= 1e4f;
            if (p[i].vx<-1e4f) p[i].vx=-1e4f;
            if (p[i].vy> 1e4f) p[i].vy= 1e4f;
            if (p[i].vy<-1e4f) p[i].vy=-1e4f;
            if (p[i].vz> 1e4f) p[i].vz= 1e4f;
            if (p[i].vz<-1e4f) p[i].vz=-1e4f;
            p[i].x+=p[i].vx*DT; p[i].y+=p[i].vy*DT; p[i].z+=p[i].vz*DT;
            p[i].lifetime-=DT;
            total_ke+=0.5*p[i].mass*(p[i].vx*p[i].vx+p[i].vy*p[i].vy+p[i].vz*p[i].vz);
        }
    }
    float result=(float)(total_ke/(PARTICLE_STEPS*PARTICLE_COUNT));
    printf("[CHILD2] %s 완료. 평균 운동에너지: %.6f\n", names[type], result);
    free(p);
    return result;
}

static float child2_particle_simulation(uint32_t seed) {
    printf("[CHILD2] 파티클 시뮬레이션 4종 시작...\n");
    float total=0;
    for (int t=0;t<PARTICLE_TYPES;t++)
        total+=simulate_effect((EffectType)t, seed^(uint32_t)(t*0x5678));
    printf("[CHILD2] 파티클 시뮬레이션 완료.\n");
    return total/PARTICLE_TYPES;
}

/* ============================================================
 * 섹션 7: [PLAY][PARENT] 물리 + AI 시야 연산
 *
 *   틱당 PHYSICS_OBJECTS개 오브젝트에 대해:
 *     - 위치/속도 업데이트 (단순 뉴턴 적분)
 *     - AI_SIGHT_RAYS개 레이캐스트 (지형 높이맵 기반)
 *   반환값: 틱 내 총 레이 히트 수 (명령서 생성에 사용)
 * ============================================================ */
static int tick_physics_ai(const TerrainMap *tmap, uint32_t *rng, int tick)
{
    int hits = 0;
    (void)tick;
    for (int obj = 0; obj < PHYSICS_OBJECTS; obj++) {
        float px = lcg_f(rng) * MAP_WIDTH;
        float py = lcg_f(rng) * MAP_HEIGHT;
        float vx = (lcg_f(rng) - 0.5f) * 2.0f;
        float vy = (lcg_f(rng) - 0.5f) * 2.0f;
        float h  = tmap->data[(int)clampf(py, 0, MAP_HEIGHT-1)]
                             [(int)clampf(px, 0, MAP_WIDTH-1)];
        vy += (9.8f * h - 0.1f * vy) * 0.016f;
        vx += (-0.1f * vx) * 0.016f;
        px += vx * 0.016f;
        py += vy * 0.016f;
        px = clampf(px, 0, MAP_WIDTH-1);
        py = clampf(py, 0, MAP_HEIGHT-1);

        for (int r = 0; r < AI_SIGHT_RAYS; r++) {
            float angle = lcg_f(rng) * 6.2832f;
            float dist  = 1.0f + lcg_f(rng) * 64.0f;
            float tx    = px + cosf(angle) * dist;
            float ty    = py + sinf(angle) * dist;
            int   ix    = (int)clampf(tx, 0, MAP_WIDTH-1);
            int   iy    = (int)clampf(ty, 0, MAP_HEIGHT-1);
            float th    = tmap->data[iy][ix];
            if (th > h + 0.1f) hits++;
        }
    }
    return hits;
}

/* ============================================================
 * 섹션 8: [PLAY][PARENT] 드로우콜 명령서 생성
 *
 *   physics_ai 결과(hits)를 받아 DRAWCALL_COUNT개 명령서 작성.
 *   노말 벡터, 조도, 패스 마스크 계산 → DrawCall 배열에 저장.
 *   가볍게 유지 (병렬화 시 GPU 대기 없이 빠르게 넘겨야 하므로)
 * ============================================================ */
static void tick_build_commands(const TerrainMap *tmap,
                                DrawCall *cmds, int hits,
                                uint32_t *rng)
{
    for (int i = 0; i < DRAWCALL_COUNT; i++) {
        float px = lcg_f(rng) * (MAP_WIDTH  - 2);
        float py = lcg_f(rng) * (MAP_HEIGHT - 2);
        int   ix = (int)px;
        int   iy = (int)py;

        float dh_dx = tmap->data[iy][ix+1] - tmap->data[iy][ix];
        float dh_dy = tmap->data[iy+1][ix] - tmap->data[iy][ix];
        float len   = sqrtf(dh_dx*dh_dx + dh_dy*dh_dy + 1.0f);

        cmds[i].object_id       = i;
        cmds[i].pos_x           = px;
        cmds[i].pos_y           = tmap->data[iy][ix];
        cmds[i].pos_z           = py;
        cmds[i].normal_x        = -dh_dx / len;
        cmds[i].normal_y        =  1.0f  / len;
        cmds[i].normal_z        = -dh_dy / len;
        cmds[i].light_intensity = clampf(
            (float)hits / (float)(PHYSICS_OBJECTS * AI_SIGHT_RAYS), 0.0f, 1.0f);
        cmds[i].pass_mask       = 1 << (i % RENDER_PASSES);
    }
}

/* ============================================================
 * 섹션 9: [PLAY][THREAD] GPU 렌더링 연산 (가정)
 *
 *   DrawCall 명령서를 READ ONLY로 받아 RENDER_PASSES 패스 실행.
 *   결과는 별도 frame_acc 버퍼에 누적 — 명령서 버퍼 불변.
 * ============================================================ */
static double tick_render_range(const DrawCall *cmds, const TerrainMap *tmap,
                                double *frame_acc, int tick,
                                int start_draw, int end_draw)
{
    double pixel_sum = 0.0;
    if (start_draw < 0) start_draw = 0;
    if (end_draw > DRAWCALL_COUNT) end_draw = DRAWCALL_COUNT;

    for (int pass = 0; pass < RENDER_PASSES; pass++) {
        for (int i = start_draw; i < end_draw; i++) {
            if (!(cmds[i].pass_mask & (1 << pass))) continue;

            float ray_ox = 512.0f, ray_oy = 2.0f, ray_oz = 512.0f;
            float ray_dx = cmds[i].pos_x - ray_ox;
            float ray_dy = cmds[i].pos_y - ray_oy;
            float ray_dz = cmds[i].pos_z - ray_oz;
            float ray_len = sqrtf(ray_dx*ray_dx + ray_dy*ray_dy + ray_dz*ray_dz);
            ray_dx /= ray_len;
            ray_dy /= ray_len;
            ray_dz /= ray_len;

            float occlusion = 0.0f;
            for (int s = 0; s < RAYMARCH_STEPS; s++) {
                float t = ray_len * (float)s / RAYMARCH_STEPS;
                float sx = ray_ox + ray_dx * t;
                float sy = ray_oy + ray_dy * t;
                float sz = ray_oz + ray_dz * t;
                int mx = (int)clampf(sx, 0, MAP_WIDTH-1);
                int mz = (int)clampf(sz, 0, MAP_HEIGHT-1);
                float terrain_h = tmap->data[mz][mx];
                if (sy < terrain_h) occlusion += 1.0f;
            }
            occlusion /= RAYMARCH_STEPS;

            float pixel = (1.0f - occlusion)
                        * cmds[i].light_intensity
                        * (1.0f + sinf((float)tick * 0.1f
                                     + cmds[i].pos_x * 0.01f) * 0.05f);

            frame_acc[i] += (double)pixel;
            pixel_sum    += (double)pixel;
        }
    }
    return pixel_sum;
}

/* ============================================================
 * 섹션 10: [PLAY] Parent + Render Thread 파이프라인
 *
 *   tick n:
 *     PARENT: 물리/AI → 명령서 생성
 *     THREAD: tick n-1 렌더링
 *
 *   명령서 버퍼는 2개를 사용한다.
 *   렌더링 중인 버퍼는 Parent가 덮어쓰지 못하도록 mutex/condition으로 보호한다.
 * ============================================================ */
typedef struct {
    const TerrainMap *tmap;
    DrawCall **cmd_buffers;
    double *frame_acc;
    int worker_id;
    int worker_count;
    int pin_cpu;
    int start_draw;
    int end_draw;
    pthread_mutex_t *mutex;
    pthread_cond_t *cond_job;
    pthread_cond_t *cond_done;
    int *stop;
    int *job_generation;
    int *job_in_progress;
    int *job_tick;
    int *job_buffer;
    int *completed_workers;
    int *rendered_ticks;
    int *buffer_busy;
    double *render_ms_sum;
    double *pixel_sum;
} PlayRenderThreadArg;

static void *play_render_thread_main(void *arg)
{
    PlayRenderThreadArg *a = (PlayRenderThreadArg *)arg;
    pin_current_thread_to_cpu(a->pin_cpu);
    int seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(a->mutex);
        while (seen_generation == *a->job_generation && !*a->stop) {
            pthread_cond_wait(a->cond_job, a->mutex);
        }
        if (*a->stop && seen_generation == *a->job_generation) {
            pthread_mutex_unlock(a->mutex);
            break;
        }

        seen_generation = *a->job_generation;
        int tick = *a->job_tick;
        int buf = *a->job_buffer;
        pthread_mutex_unlock(a->mutex);

        double t = now_ms();
        double pixels = tick_render_range(a->cmd_buffers[buf], a->tmap, a->frame_acc,
                                          tick, a->start_draw, a->end_draw);
        double elapsed = now_ms() - t;

        pthread_mutex_lock(a->mutex);
        *a->render_ms_sum += elapsed;
        *a->pixel_sum += pixels;
        (*a->completed_workers)++;
        if (*a->completed_workers == a->worker_count) {
            a->buffer_busy[buf] = 0;
            *a->job_in_progress = 0;
            (*a->rendered_ticks)++;
            pthread_cond_broadcast(a->cond_done);
        }
        pthread_mutex_unlock(a->mutex);
    }

    return NULL;
}

static void play_threaded_pipeline_simulation(const TerrainMap *tmap,
                                              PerfMetrics *m)
{
    const int worker_count = 1;
    printf("[PLAY] Parent + 단일 Render Thread 파이프라인 시작...\n");
    printf("[PLAY] 총 %d틱 | 물리 %d오브젝트 | 시야 %d레이 | 렌더 %d패스 | 드로우콜 %d\n",
           PLAY_TICKS, PHYSICS_OBJECTS, AI_SIGHT_RAYS, RENDER_PASSES, DRAWCALL_COUNT);

    uint32_t rng = FIXED_SEED ^ 0xCAFEF00D;
    DrawCall *cmd_buffers[2];
    cmd_buffers[0] = malloc(sizeof(DrawCall) * DRAWCALL_COUNT);
    cmd_buffers[1] = malloc(sizeof(DrawCall) * DRAWCALL_COUNT);
    double *frame_acc = calloc(DRAWCALL_COUNT, sizeof(double));
    if (!cmd_buffers[0] || !cmd_buffers[1] || !frame_acc) {
        free(cmd_buffers[0]);
        free(cmd_buffers[1]);
        free(frame_acc);
        return;
    }

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond_job = PTHREAD_COND_INITIALIZER;
    pthread_cond_t cond_done = PTHREAD_COND_INITIALIZER;
    int stop = 0;
    int job_generation = 0;
    int job_in_progress = 0;
    int job_tick = -1;
    int job_buffer = 0;
    int completed_workers = 0;
    int rendered_ticks = 0;
    int buffer_busy[2] = {0, 0};
    double render_ms_sum = 0.0;
    double pixel_sum = 0.0;

    pthread_t render_thread;
    PlayRenderThreadArg arg;
    memset(&arg, 0, sizeof(arg));
    arg.tmap = tmap;
    arg.cmd_buffers = cmd_buffers;
    arg.frame_acc = frame_acc;
    arg.worker_id = 0;
    arg.worker_count = worker_count;
    arg.pin_cpu = role_cpu(1);
    arg.start_draw = 0;
    arg.end_draw = DRAWCALL_COUNT;
    arg.mutex = &mutex;
    arg.cond_job = &cond_job;
    arg.cond_done = &cond_done;
    arg.stop = &stop;
    arg.job_generation = &job_generation;
    arg.job_in_progress = &job_in_progress;
    arg.job_tick = &job_tick;
    arg.job_buffer = &job_buffer;
    arg.completed_workers = &completed_workers;
    arg.rendered_ticks = &rendered_ticks;
    arg.buffer_busy = buffer_busy;
    arg.render_ms_sum = &render_ms_sum;
    arg.pixel_sum = &pixel_sum;

    double create_start = now_ms();
    int rc = pthread_create(&render_thread, NULL, play_render_thread_main, &arg);
    m->play_thread_create_ms = now_ms() - create_start;
    if (rc != 0) {
        fprintf(stderr, "play render pthread_create failed: %d\n", rc);
        free(cmd_buffers[0]);
        free(cmd_buffers[1]);
        free(frame_acc);
        return;
    }

    double t_physics = 0.0;
    double t_cmd = 0.0;
    int progress_step = PLAY_TICKS >= 10 ? PLAY_TICKS / 10 : 1;

    for (int tick = 0; tick < PLAY_TICKS; tick++) {
        int buf = tick % 2;

        pthread_mutex_lock(&mutex);
        while (buffer_busy[buf]) {
            pthread_cond_wait(&cond_done, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        double t = now_ms();
        int hits = tick_physics_ai(tmap, &rng, tick);
        t_physics += now_ms() - t;

        t = now_ms();
        tick_build_commands(tmap, cmd_buffers[buf], hits, &rng);
        t_cmd += now_ms() - t;

        pthread_mutex_lock(&mutex);
        while (job_in_progress) {
            pthread_cond_wait(&cond_done, &mutex);
        }
        buffer_busy[buf] = 1;
        job_tick = tick;
        job_buffer = buf;
        completed_workers = 0;
        job_in_progress = 1;
        job_generation++;
        pthread_cond_broadcast(&cond_job);
        pthread_mutex_unlock(&mutex);

        if ((tick + 1) % progress_step == 0) {
            printf("[PLAY] tick %3d/%d | Parent 누적 %.1fms | Render Thread 진행 중\n",
                   tick + 1, PLAY_TICKS, t_physics + t_cmd);
        }
    }

    pthread_mutex_lock(&mutex);
    while (rendered_ticks < PLAY_TICKS) {
        pthread_cond_wait(&cond_done, &mutex);
    }
    stop = 1;
    pthread_cond_broadcast(&cond_job);
    pthread_mutex_unlock(&mutex);
    pthread_join(render_thread, NULL);

    printf("[PLAY] 완료\n");
    printf("[PLAY] 물리/AI:       %8.2f ms (Parent)\n", t_physics);
    printf("[PLAY] 명령서 생성:   %8.2f ms (Parent)\n", t_cmd);
    printf("[PLAY] GPU 렌더링:    %8.2f ms (Render Thread 실제 계산 누적)\n", render_ms_sum);
    printf("[PLAY] 평균 픽셀 밝기: %.6f\n", pixel_sum / (PLAY_TICKS * DRAWCALL_COUNT));

    m->play_ticks = PLAY_TICKS;
    m->play_physics_ms = t_physics;
    m->play_cmd_ms = t_cmd;
    m->play_render_ms = render_ms_sum;
    m->play_roundrobin_ms = t_physics + t_cmd + render_ms_sum;

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_job);
    pthread_cond_destroy(&cond_done);
    free(cmd_buffers[0]);
    free(cmd_buffers[1]);
    free(frame_acc);
}

/* ============================================================
 * 병렬 실험 계측 보조 함수
 * ============================================================ */

typedef struct {
    unsigned long long total;
    unsigned long long idle;
} CpuCoreSnap;

typedef struct {
    int count;
    CpuCoreSnap core[128];
} CpuSnapshot;

typedef struct {
    double user_ms;
    double sys_ms;
    long voluntary;
    long nonvoluntary;
} UsageSnap;

static long get_mem_kb(const char *key)
{
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return -1;
    char line[256];
    long value = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, key, strlen(key)) == 0) {
            if (sscanf(line + strlen(key), ": %ld", &value) != 1) value = -1;
            break;
        }
    }
    fclose(fp);
    return value;
}

static int get_core_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

static UsageSnap get_usage_snap(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    UsageSnap s;
    s.user_ms = ru.ru_utime.tv_sec * 1000.0 + ru.ru_utime.tv_usec / 1000.0;
    s.sys_ms  = ru.ru_stime.tv_sec * 1000.0 + ru.ru_stime.tv_usec / 1000.0;
    s.voluntary = ru.ru_nvcsw;
    s.nonvoluntary = ru.ru_nivcsw;
    return s;
}

static void read_cpu_snapshot(CpuSnapshot *snap)
{
    memset(snap, 0, sizeof(*snap));
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return;

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        int id;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu",
                   &id, &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) == 9) {
            if (id >= 0 && id < 128) {
                unsigned long long idle_all = idle + iowait;
                unsigned long long non_idle = user + nice + system + irq + softirq + steal;
                snap->core[id].idle = idle_all;
                snap->core[id].total = idle_all + non_idle;
                if (id + 1 > snap->count) snap->count = id + 1;
            }
        }
    }
    fclose(fp);
}

static double cpu_usage_between(const CpuSnapshot *a, const CpuSnapshot *b, int cpu)
{
    if (cpu < 0 || cpu >= a->count || cpu >= b->count) return -1.0;
    unsigned long long total_delta = b->core[cpu].total - a->core[cpu].total;
    unsigned long long idle_delta  = b->core[cpu].idle  - a->core[cpu].idle;
    if (total_delta == 0) return -1.0;
    return ((double)(total_delta - idle_delta) / (double)total_delta) * 100.0;
}

static double cpu_usage_avg_range(const CpuSnapshot *a, const CpuSnapshot *b, int start, int end_exclusive)
{
    double sum = 0.0;
    int count = 0;
    int max_count = a->count < b->count ? a->count : b->count;
    if (start < 0) start = 0;
    if (end_exclusive > max_count) end_exclusive = max_count;
    for (int i = start; i < end_exclusive; i++) {
        double u = cpu_usage_between(a, b, i);
        if (u >= 0.0) {
            sum += u;
            count++;
        }
    }
    return count > 0 ? sum / (double)count : -1.0;
}

static void print_percent_or_na(double v)
{
    if (v < 0.0) printf("N/A");
    else printf("%.1f%%", v);
}

static double calc_speedup(double baseline_ms, double wall_ms)
{
    return (baseline_ms > 0.0 && wall_ms > 0.0) ? baseline_ms / wall_ms : 0.0;
}

static double calc_efficiency(double speedup, int n)
{
    return (n > 0) ? speedup / (double)n * 100.0 : 0.0;
}

static double calc_p_effective(double speedup, int n)
{
    if (speedup <= 1.0 || n <= 1) return 0.0;
    return (1.0 - 1.0 / speedup) / (1.0 - 1.0 / (double)n);
}

static double calc_amdahl(double p, int n)
{
    if (n <= 0) return 0.0;
    return 1.0 / ((1.0 - p) + p / (double)n);
}

static TerrainMap *alloc_terrain_map(void)
{
    TerrainMap *tmap = calloc(1, sizeof(TerrainMap));
    if (!tmap) return NULL;

    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        int cx = i % CHUNKS_PER_ROW;
        int cy = i / CHUNKS_PER_ROW;
        tmap->chunks[i].chunk_id = i;
        tmap->chunks[i].chunk_x = cx;
        tmap->chunks[i].chunk_y = cy;
        tmap->chunks[i].pixel_x_start = cx * CHUNK_SIZE;
        tmap->chunks[i].pixel_y_start = cy * CHUNK_SIZE;
        tmap->chunks[i].height_data = malloc(sizeof(float) * CHUNK_SIZE * CHUNK_SIZE);
        if (!tmap->chunks[i].height_data) {
            for (int j = 0; j < i; j++) free(tmap->chunks[j].height_data);
            free(tmap);
            return NULL;
        }
    }
    return tmap;
}

static void free_terrain_map(TerrainMap *tmap)
{
    if (!tmap) return;
    for (int i = 0; i < TOTAL_CHUNKS; i++) free(tmap->chunks[i].height_data);
    free(tmap);
}
/* ============================================================
 * 나 실험 — Parent + 단일 Thread
 * - Thread: 지형 생성 전체 담당(noise→merge→blur→erosion→hist/integrity)
 * - Parent: Thread와 동시에 조도 계산 수행
 * - fork()는 사용하지 않고 pthread_create() 비용을 측정한다.
 * ============================================================ */

typedef struct {
    TerrainMap *tmap;
    PerfMetrics *m;
} TerrainFullThreadArg;

static void *terrain_full_thread_main(void *arg)
{
    pin_current_thread_to_cpu(role_cpu(1));
    TerrainFullThreadArg *a = (TerrainFullThreadArg *)arg;
    TerrainMap *tmap = a->tmap;
    PerfMetrics *m = a->m;
    double t;

    init_noise_params(&tmap->params);

    printf("\n[WORKER-THREAD] 단계1: 노이즈 연산 전체(%d청크)...\n", TOTAL_CHUNKS);
    t = now_ms();
    for (int ci = 0; ci < TOTAL_CHUNKS; ci++) {
        noise_compute_chunk(&tmap->chunks[ci], &tmap->params);
        if ((ci + 1) % 64 == 0) {
            printf("[WORKER-THREAD] %d/%d (%.0f%%)\n",
                   ci + 1, TOTAL_CHUNKS, (ci + 1) * 100.0 / TOTAL_CHUNKS);
        }
    }
    m->thread_noise_ms = now_ms() - t;

    printf("[WORKER-THREAD] 단계2: 청크 병합...\n");
    t = now_ms();
    terrain_merge(tmap);
    m->thread_merge_ms = now_ms() - t;

    printf("[WORKER-THREAD] 단계3: 경계 블러...\n");
    t = now_ms();
    terrain_blur(tmap);
    m->thread_blur_ms = now_ms() - t;

    printf("[WORKER-THREAD] 단계4: 수력 침식...\n");
    t = now_ms();
    terrain_erosion(tmap);
    m->thread_erosion_ms = now_ms() - t;

    printf("[WORKER-THREAD] 단계5: 히스토그램 + 무결성 검사...\n");
    t = now_ms();
    terrain_histogram(tmap);
    m->integrity_fails = terrain_integrity(tmap);
    m->thread_integrity_ms = now_ms() - t;

    m->thread_total_ms = m->thread_noise_ms + m->thread_merge_ms +
                         m->thread_blur_ms + m->thread_erosion_ms +
                         m->thread_integrity_ms;

    printf("[WORKER-THREAD] 지형 생성 완료: %.2f ms | 무결성 실패: %d\n\n",
           m->thread_total_ms, m->integrity_fails);
    return NULL;
}

static void print_na_report(const PerfMetrics *m,
                            double baseline_ms,
                            double setting_wall_ms,
                            double transition_ms,
                            double play_wall_ms,
                            double total_wall_ms,
                            double pthread_create_ms,
                            const UsageSnap *setting_usage,
                            const UsageSnap *play_usage,
                            const UsageSnap *total_usage,
                            long setting_rss_kb,
                            long setting_peak_kb,
                            long play_rss_kb,
                            long play_peak_kb,
                            const CpuSnapshot *cpu_setting_start,
                            const CpuSnapshot *cpu_setting_end,
                            const CpuSnapshot *cpu_play_start,
                            const CpuSnapshot *cpu_play_end)
{
    const int n_setting = 2;
    const int n_play = 2;
    const int n_total_reference = 2;

    double setting_parallel_seq = m->thread_total_ms + m->parent_lighting_ms +
                                  m->child1_ai_ms + m->child2_particle_ms;
    double parent_side_ms = m->parent_lighting_ms + m->child1_ai_ms + m->child2_particle_ms;
    double setting_parallel_wall = m->thread_total_ms > parent_side_ms ?
                                   m->thread_total_ms : parent_side_ms;
    double setting_parallel_speedup = setting_parallel_wall > 0.0 ?
                                      setting_parallel_seq / setting_parallel_wall : 0.0;
    double setting_seq = setting_parallel_seq;
    double setting_speedup = setting_wall_ms > 0.0 ? setting_seq / setting_wall_ms : 0.0;
    double play_seq = m->play_physics_ms + m->play_cmd_ms + m->play_render_ms;
    double play_speedup = play_wall_ms > 0.0 ? play_seq / play_wall_ms : 0.0;
    double total_speedup = calc_speedup(baseline_ms, total_wall_ms);
    double p_eff = calc_p_effective(total_speedup, n_total_reference);
    double s_theory = calc_amdahl(p_eff, n_total_reference);
    double cpu_util = total_wall_ms > 0.0 ?
        (total_usage->user_ms + total_usage->sys_ms) / total_wall_ms * 100.0 : 0.0;

    printf("\n===========================================================\n");
    printf("=== 게임 월드 병렬 실험 — 나 ===\n");
    printf("시드: 0x%08X (고정) | 맵: %dx%d | 청크: %d개\n", FIXED_SEED, MAP_WIDTH, MAP_HEIGHT, TOTAL_CHUNKS);
    printf("\n[CORE] 시스템 코어 수: %d개\n", get_core_count());
    printf("[PIN] 적용 시도: PARENT → cpu0 / WORKER·RENDER → cpu%d\n", role_cpu(1));

    printf("\n===========================================================\n");
    printf("[실험 구성]\n");
    printf("===========================================================\n");
    printf("  구성: Parent + 단일 Thread\n");
    printf("  실행 주체 수: Parent 1 + Thread 1\n");
    printf("  active core 수 N_setting: %d\n", n_setting);
    printf("  active core 수 N_play:    %d\n", n_play);
    printf("  N 계산 근거: 세팅 중 Parent는 조도 계산, Thread는 지형 생성 전체를 동시에 수행. 플레이 중 Parent는 현재 tick 물리/명령서, Thread는 이전 tick 렌더링을 수행.\n");
    printf("  비교 목적: fork() 비용 vs pthread_create() 비용 중 Thread 측정\n");

    printf("\n===========================================================\n");
    printf("[세팅 단계]\n");
    printf("===========================================================\n");
    printf("  [WORKER] 지형 생성 전체:  %10.2f ms  (noise→merge→blur→erosion→hist/integrity)\n", m->thread_total_ms);
    printf("    노이즈:                 %10.2f ms\n", m->thread_noise_ms);
    printf("    병합:                   %10.2f ms  (완료 후 BARRIER POINT 1 → 블러 시작 가능)\n", m->thread_merge_ms);
    printf("    블러:                   %10.2f ms  (BARRIER POINT 1 이후 시작, 완료 후 BARRIER POINT 2 → 침식 시작 가능)\n", m->thread_blur_ms);
    printf("    침식:                   %10.2f ms  (BARRIER POINT 2 이후 시작, 완료 후 BARRIER POINT 3 → hist/integrity 시작 가능)\n", m->thread_erosion_ms);
    printf("    히스토그램/무결성:       %10.2f ms  (BARRIER POINT 3 이후 시작)\n", m->thread_integrity_ms);
    printf("  [PARENT] 조도 계산:       %10.2f ms  (WORKER와 동시 실행)\n", m->parent_lighting_ms);
    printf("  [AI] 몬스터 AI:           %10.2f ms  (순차 실행, 계산식 유지)\n", m->child1_ai_ms);
    printf("  [PARTICLE] 파티클 4종:    %10.2f ms  (순차 실행, 계산식 유지)\n", m->child2_particle_ms);
    printf("  세팅 병렬 구간 순차 환산: %10.2f ms  (= 지형 생성 + 조도 + AI + 파티클)\n", setting_parallel_seq);
    printf("  세팅 병렬 구간 Wall:      %10.2f ms  (= max(지형 생성, 조도+AI+파티클))\n", setting_parallel_wall);
    printf("  세팅 병렬 구간 Speedup:   %10.2fx\n", setting_parallel_speedup);
    printf("  세팅 단계 Wall Time:      %10.2f ms\n", setting_wall_ms);
    printf("  세팅 전체 순차 환산:      %10.2f ms  (= 지형 생성 + 조도 + AI + 파티클)\n", setting_seq);
    printf("  세팅 Speedup:             %10.2fx\n", setting_speedup);
    printf("  세팅 Efficiency:          %10.1f%%  (= Speedup / N_setting)\n", calc_efficiency(setting_speedup, n_setting));

    printf("\n[프로세스/스레드 생성]\n");
    printf("  fork() × 1:               N/A\n");
    printf("  pthread_create() × 1:     %10.3f ms\n", pthread_create_ms);
    printf("  생성 총 오버헤드:          %10.3f ms\n", pthread_create_ms);

    printf("\n===========================================================\n");
    printf("[세팅→플레이 전환]\n");
    printf("===========================================================\n");
    printf("  추가 pthread_create():    %10.3f ms  (play render thread)\n", m->play_thread_create_ms);
    printf("  전환 총 시간:             %10.3f ms\n", transition_ms + m->play_thread_create_ms);

    printf("\n===========================================================\n");
    printf("[플레이 단계] %d틱\n", PLAY_TICKS);
    printf("===========================================================\n");
    printf("  [PARENT] 물리/AI 연산:    %10.2f ms\n", m->play_physics_ms);
    printf("  [PARENT] 명령서 생성:     %10.2f ms\n", m->play_cmd_ms);
    printf("  [WORKER] GPU 렌더링:      %10.2f ms  (단일 Render Thread, 이전 tick 렌더링)\n", m->play_render_ms);
    printf("  플레이 Wall Time:         %10.2f ms\n", play_wall_ms);
    printf("  플레이 순차 환산:         %10.2f ms\n", play_seq);
    printf("  플레이 Speedup:           %10.2fx\n", play_speedup);
    printf("  플레이 Efficiency:        %10.1f%%  (= Speedup / N_play)\n", calc_efficiency(play_speedup, n_play));

    printf("\n===========================================================\n");
    printf("  총 Wall Time:             %10.2f ms\n", total_wall_ms);
    if (baseline_ms > 0.0) {
        printf("  총 Speedup:               %10.2fx  (기준: Baseline %.2f ms)\n", total_speedup, baseline_ms);
        printf("  총 Efficiency:            %10.1f%%  (= Speedup / 대표 N_setting)\n", calc_efficiency(total_speedup, n_total_reference));
    } else {
        printf("  총 Speedup:               N/A  (Baseline Wall Time 인자 필요)\n");
        printf("  총 Efficiency:            N/A\n");
    }
    printf("===========================================================\n");
    printf("[Amdahl's Law 검증]\n");
    printf("  active core 수 N:          %d  (총괄 비교용 대표 N_setting)\n", n_total_reference);
    printf("  병렬화 비율 P_effective:   %10.1f%%\n", p_eff * 100.0);
    printf("  이론 Speedup:              %10.2fx\n", s_theory);
    printf("  실측 Speedup:              %10.2fx\n", total_speedup);
    printf("  차이:                      %10.2fx\n", fabs(s_theory - total_speedup));
    printf("===========================================================\n");
    printf("[CPU 성능 지표]\n");
    printf("  CPU user 시간:             %10.2f ms\n", total_usage->user_ms);
    printf("  CPU sys  시간:             %10.2f ms\n", total_usage->sys_ms);
    printf("  CPU 활용률:                %10.1f%%\n", cpu_util);
    printf("===========================================================\n");
    printf("[세팅 단계 코어별]        [플레이 단계 코어별]\n");
    printf("  cpu0 (PARENT):  ");
    print_percent_or_na(cpu_usage_between(cpu_setting_start, cpu_setting_end, 0));
    printf("    cpu0 (PARENT):  ");
    print_percent_or_na(cpu_usage_between(cpu_play_start, cpu_play_end, 0));
    printf("\n  cpu1 (WORKER):  ");
    print_percent_or_na(cpu_usage_between(cpu_setting_start, cpu_setting_end, 1));
    printf("    cpu1 (RENDER):  ");
    print_percent_or_na(cpu_usage_between(cpu_play_start, cpu_play_end, 1));
    printf("\n  cpu2~:          ");
    print_percent_or_na(cpu_usage_avg_range(cpu_setting_start, cpu_setting_end, 2, 128));
    printf("    cpu2~:          ");
    print_percent_or_na(cpu_usage_avg_range(cpu_play_start, cpu_play_end, 2, 128));
    printf("\n");
    printf("===========================================================\n");
    printf("[컨텍스트 스위칭]\n");
    printf("               voluntary    nonvoluntary\n");
    printf("  세팅 단계:    %8ld 회  %8ld 회\n", setting_usage->voluntary, setting_usage->nonvoluntary);
    printf("  플레이 단계:  %8ld 회  %8ld 회\n", play_usage->voluntary, play_usage->nonvoluntary);
    printf("  Baseline 대비: 별도 Baseline rusage 측정값을 보고서에서 비교\n");
    printf("===========================================================\n");
    printf("[IPC 오버헤드]\n");
    printf("  pipe/shared memory/none:  none (Thread 구조, 주소 공간 공유)\n");
    printf("  총 오버헤드:              N/A\n");
    printf("===========================================================\n");
    printf("[메모리 사용량]\n");
    printf("  세팅 완료 후 RSS:          %10ld KB\n", setting_rss_kb);
    printf("  세팅 완료 후 Peak:         %10ld KB\n", setting_peak_kb);
    printf("  플레이 완료 후 RSS:        %10ld KB\n", play_rss_kb);
    printf("  플레이 완료 후 Peak:       %10ld KB\n", play_peak_kb);
    printf("===========================================================\n");
    printf("[결과 검증 — Baseline과 동일해야 함]\n");
    printf("  평균 조도:                 %.6f\n", m->lighting_result);
    printf("  AI 최고점수:               %.4f\n", m->ai_best_score);
    printf("  파티클 에너지:             %.6f\n", m->particle_final_energy);
    printf("  무결성 실패:               %d\n", m->integrity_fails);
    printf("===========================================================\n");
}

int main(int argc, char **argv)
{
    pin_current_thread_to_cpu(role_cpu(0));
    double baseline_ms = 0.0;
    if (argc >= 2) baseline_ms = atof(argv[1]);

    printf("=== 게임 월드 병렬 실험 — 나: Parent + 단일 Thread ===\n");
    printf("Baseline Wall Time 인자: %.2f ms%s\n", baseline_ms, baseline_ms > 0.0 ? "" : " (미입력)");
    printf("컴파일 예: gcc -O2 -o na_thread1_play_fixed na_thread1_play_fixed.c -lm -lpthread\n\n");

    TerrainMap *tmap = alloc_terrain_map();
    if (!tmap) {
        perror("alloc_terrain_map");
        return 1;
    }

    PerfMetrics m;
    memset(&m, 0, sizeof(m));

    CpuSnapshot cpu_setting_start, cpu_setting_end, cpu_play_start, cpu_play_end;
    UsageSnap usage_start, usage_after_setting, usage_after_play;

    read_cpu_snapshot(&cpu_setting_start);
    usage_start = get_usage_snap();

    double total_start = now_ms();
    double setting_start = now_ms();

    pthread_t terrain_thread;
    TerrainFullThreadArg arg;
    arg.tmap = tmap;
    arg.m = &m;

    double t = now_ms();
    int rc = pthread_create(&terrain_thread, NULL, terrain_full_thread_main, &arg);
    double pthread_create_ms = now_ms() - t;
    if (rc != 0) {
        fprintf(stderr, "pthread_create failed: %d\n", rc);
        free_terrain_map(tmap);
        return 1;
    }

    t = now_ms();
    m.lighting_result = parent_lighting_simulation(FIXED_SEED);
    m.parent_lighting_ms = now_ms() - t;
    printf("[PARENT] 조도 계산 완료: %.2f ms\n\n", m.parent_lighting_ms);

    t = now_ms();
    m.ai_best_score = child1_monster_ai(FIXED_SEED);
    m.child1_ai_ms = now_ms() - t;

    t = now_ms();
    m.particle_final_energy = child2_particle_simulation(FIXED_SEED);
    m.child2_particle_ms = now_ms() - t;

    pthread_join(terrain_thread, NULL);

    double setting_wall_ms = now_ms() - setting_start;
    read_cpu_snapshot(&cpu_setting_end);
    usage_after_setting = get_usage_snap();
    long setting_rss_kb = get_mem_kb("VmRSS");
    long setting_peak_kb = get_mem_kb("VmHWM");

    double transition_start = now_ms();
    double transition_ms = now_ms() - transition_start;

    read_cpu_snapshot(&cpu_play_start);
    double play_start = now_ms();
    play_threaded_pipeline_simulation(tmap, &m);
    double play_wall_ms = now_ms() - play_start;
    m.play_roundrobin_ms = play_wall_ms;
    read_cpu_snapshot(&cpu_play_end);
    usage_after_play = get_usage_snap();
    long play_rss_kb = get_mem_kb("VmRSS");
    long play_peak_kb = get_mem_kb("VmHWM");

    double total_wall_ms = now_ms() - total_start;
    m.total_ms = total_wall_ms;

    UsageSnap setting_usage = {
        usage_after_setting.user_ms - usage_start.user_ms,
        usage_after_setting.sys_ms - usage_start.sys_ms,
        usage_after_setting.voluntary - usage_start.voluntary,
        usage_after_setting.nonvoluntary - usage_start.nonvoluntary
    };
    UsageSnap play_usage = {
        usage_after_play.user_ms - usage_after_setting.user_ms,
        usage_after_play.sys_ms - usage_after_setting.sys_ms,
        usage_after_play.voluntary - usage_after_setting.voluntary,
        usage_after_play.nonvoluntary - usage_after_setting.nonvoluntary
    };
    UsageSnap total_usage = {
        usage_after_play.user_ms - usage_start.user_ms,
        usage_after_play.sys_ms - usage_start.sys_ms,
        usage_after_play.voluntary - usage_start.voluntary,
        usage_after_play.nonvoluntary - usage_start.nonvoluntary
    };

    print_na_report(&m,
                    baseline_ms,
                    setting_wall_ms,
                    transition_ms,
                    play_wall_ms,
                    total_wall_ms,
                    pthread_create_ms,
                    &setting_usage,
                    &play_usage,
                    &total_usage,
                    setting_rss_kb,
                    setting_peak_kb,
                    play_rss_kb,
                    play_peak_kb,
                    &cpu_setting_start,
                    &cpu_setting_end,
                    &cpu_play_start,
                    &cpu_play_end);

    free_terrain_map(tmap);
    return 0;
}
