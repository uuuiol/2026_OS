/*
 * ============================================================
 *  game_world_parallel_동기화 문제 유도 및 동기화 방법 적용
 *  뮤텍스/세마포어/모니터 각각 적용시켰으며 적용시에 다른코드 및 barrier를 주석처리후 실행 
 *  이 파일은 Barrier 1, 2, 3에 semaphore 방식 동기화를 적용한 버전 
 * ============================================================
 *
 *  시스템 개요:
 *    [세팅 단계] THREAD×4 + PARENT + CHILD1 + CHILD2 병렬 실행
 *    [플레이 단계] CHILD 종료 후 PARENT가 스레드 2개 추가 생성
 *                  → PARENT(물리/AI) + THREAD×6(GPU렌더) 병렬 처리
 *
 *  태스크 구성:
 *    [THREAD×4] 지형 맵 생성 : 노이즈 → 블러 → 침식 → 병합
 *    [PARENT]   조도 계산    : 몬테카를로 광선 추적 (Thread와 동시)
 *    [CHILD1]   몬스터 AI    : 시뮬레이티드 어닐링 행동트리 최적화
 *    [CHILD2]   파티클 시뮬  : 폭발/물보라/용암/연기 4종
 *    [PLAY]     틱 파이프라인: PARENT 물리/AI → 명령서 → THREAD×6 GPU렌더
 *
 *  병렬 구조:
 *    세팅: pthread×N + fork×K 동시 시작, 남은 CHILD 작업은 PARENT가 수행
 *    플레이: waitpid 후 CHILD 종료 확인 → pthread×2 추가 생성
 *            PARENT 틱루프(물리/AI+명령서) ↔ THREAD×6 틱루프(GPU렌더) 병렬
 *            더블버퍼: PARENT 현재 틱 명령서 ↔ THREAD 이전 틱 명령서
 
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <pthread.h>
#include <semaphore.h>

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
static pthread_mutex_t custom_b1_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t custom_b1_cond = PTHREAD_COND_INITIALIZER;
static int custom_b1_count = 0;

static pthread_mutex_t custom_b2_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t custom_b2_cond = PTHREAD_COND_INITIALIZER;
static int custom_b2_count = 0;

static pthread_mutex_t custom_b3_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t custom_b3_cond = PTHREAD_COND_INITIALIZER;
static int custom_b3_count = 0;

/* 커스텀 배리어를 위한 세마포어 및 카운터 선언 (BARRIER 1, 2, 3) */
static sem_t sem_b1_mutex;
static sem_t sem_b1_turnstile;
static int sem_b1_count = 0;

static sem_t sem_b2_mutex;
static sem_t sem_b2_turnstile;
static int sem_b2_count = 0;

static sem_t sem_b3_mutex;
static sem_t sem_b3_turnstile;
static int sem_b3_count = 0;


/* ============================================================
 * 커스텀 BARRIER 구현 3: 세마포어 방식
 * - mutex 역할 세마포어로 count를 보호한다.
 * - turnstile 세마포어를 이용해 마지막 스레드 도착 전까지 모두 대기시킨다.
 * - 이 프로그램에서는 Barrier 1, 2, 3이 각각 1회만 사용되므로 one-shot barrier로 충분하다.
 * ============================================================ */
static void semaphore_barrier_wait(sem_t *mutex_sem,
                                   sem_t *turnstile_sem,
                                   int *count,
                                   int target_count)
{
    sem_wait(mutex_sem);
    (*count)++;
    if (*count == target_count) {
        sem_post(turnstile_sem);
    }
    sem_post(mutex_sem);

    sem_wait(turnstile_sem);
    sem_post(turnstile_sem);
}


/* 실험용 최대값/기본값
 * 실행 시 인자로 변경 가능:
 *   --threads N       : 세팅 단계 지형 생성 pthread 수, 1~16
 *   --children N      : fork child 수, 0~2 (남은 CHILD 작업은 PARENT가 수행)
 *                       0=child 없음, 1=CHILD1(AI)만, 2=CHILD1+CHILD2
 *   --play-threads N  : 플레이 단계 GPU 렌더 pthread 총 수, 1~32
 *                       생략하면 threads+2로 설정
 */
#define DEFAULT_NUM_THREADS         4
#define DEFAULT_CHILD_COUNT         2
#define DEFAULT_PLAY_EXTRA_THREADS  2
#define MAX_SETUP_THREADS           16
#define MAX_PLAY_THREADS            32

static int g_num_threads = DEFAULT_NUM_THREADS;
static int g_child_count = DEFAULT_CHILD_COUNT;
static int g_play_render_threads = DEFAULT_NUM_THREADS + DEFAULT_PLAY_EXTRA_THREADS;

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
#define LIGHT_SAMPLES   340000000L

/* [CHILD1] 몬스터 AI 파라미터 */
#define AI_MONSTERS     25
#define AI_WEIGHTS      16
#define AI_ANNEAL_STEPS 325
#define AI_COMBAT_SIM   1410
#define AI_COMBAT_TICKS 300

/* [CHILD2] 파티클 파라미터 */
#define PARTICLE_TYPES  4
#define PARTICLE_COUNT  400
#define PARTICLE_STEPS  5500

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

typedef struct {
    long long user, nice, system, idle, iowait, irq, softirq;
} CoreStat;

typedef struct {
    CoreStat cores[16];
    int num_cores;
} CpuSnapshot;

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
    double setting_wall_ms;     /* 세팅 병렬 구간 벽시계 */
    /* 플레이 단계 */
    double play_wall_ms;        /* 플레이 병렬 구간 벽시계 */
    double play_roundrobin_ms;
    int    play_ticks;
    double play_physics_ms;     /* PARENT 물리/AI 누적 */
    double play_cmd_ms;         /* PARENT 명령서 생성 누적 */
    double play_render_ms;      /* THREAD GPU렌더 누적 (스레드별 최대) */
    /* 전체 */
    double total_ms;
    /* CPU 성능 지표 */
    double cpu_user_ms;
    double cpu_sys_ms;
    /* 코어별 CPU 활용률 */
    double setup_core_util[16];
    double play_core_util[16];
    int    num_cores;
    /* IPC */
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

/* ── 세팅 단계 스레드 인자 ── */
typedef struct {
    int               thread_id;
    TerrainMap       *tmap;
    pthread_barrier_t *barrier;
    double            elapsed_noise_ms;
    double            elapsed_blur_ms;
    double            elapsed_erosion_ms;
    double            elapsed_merge_ms;
    /* 플레이 단계: 세팅 완료 후 바로 렌더 루프 진입 */
    struct PlayShared *shared;  /* main이 CHILD 종료 후 설정 */
    pthread_barrier_t *play_barrier;   /* 세팅→플레이 전환 동기화 */
    pthread_barrier_t *verify_barrier; /* 클램핑 완료 → 무결성 검사 허용 */
} ThreadArg;

/* ── 플레이 단계 더블버퍼 공유 상태 ── */
typedef struct PlayShared {
    /* 더블버퍼: buf[0], buf[1] 교대 사용 */
    DrawCall     *buf[2];
    double       *frame_acc[2];
    /* 동기화 */
    pthread_mutex_t  mutex; //모니터랑 뮤텍스락 설정용 변수 
    pthread_cond_t   cond_render;   /* PARENT → THREAD: 새 명령서 준비됨 */
    pthread_cond_t   cond_physics;  /* THREAD → PARENT: 렌더 완료, 버퍼 반환 */
    int              ready_buf;     /* THREAD가 렌더할 버퍼 인덱스 (-1=없음) 모니터의 상태 변수 역할 */
    int              done_count;    /* 현재 틱 렌더 완료한 스레드 수 */
    int              finished;      /* 플레이 종료 플래그 */
    int              total_ticks;
    int              current_tick;
    const TerrainMap *tmap;
    /* 결과 수집 */
    double           render_ms_per_thread[MAX_PLAY_THREADS];
    double           pixel_grand;
} PlayShared;

/* ── 플레이 단계 렌더 스레드 인자 ── */
typedef struct {
    int         thread_id;
    PlayShared *shared;
} RenderThreadArg;

/* ============================================================
 * 섹션 2: 유틸리티
 * ============================================================ */

static void cpu_snapshot(CpuSnapshot *snap) {
    snap->num_cores = 0;
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f) && snap->num_cores < 16) {
        if (strncmp(line, "cpu", 3) != 0) break;
        if (line[3] == ' ') continue;  /* "cpu " 전체 합산 줄 건너뜀 */
        long long u, n, s, id, io, irq, sirq;
        if (sscanf(line, "cpu%*d %lld %lld %lld %lld %lld %lld %lld",
                   &u, &n, &s, &id, &io, &irq, &sirq) == 7) {
            int idx = snap->num_cores++;
            snap->cores[idx].user    = u;
            snap->cores[idx].nice    = n;
            snap->cores[idx].system  = s;
            snap->cores[idx].idle    = id;
            snap->cores[idx].iowait  = io;
            snap->cores[idx].irq     = irq;
            snap->cores[idx].softirq = sirq;
        }
    }
    fclose(f);
}

static void cpu_calc_util(const CpuSnapshot *before,
                           const CpuSnapshot *after,
                           double *util_out,
                           int *num_cores_out) {
    int n = before->num_cores < after->num_cores
            ? before->num_cores : after->num_cores;
    *num_cores_out = n;
    for (int i = 0; i < n; i++) {
        long long du  = after->cores[i].user    - before->cores[i].user;
        long long dn  = after->cores[i].nice    - before->cores[i].nice;
        long long ds  = after->cores[i].system  - before->cores[i].system;
        long long di  = after->cores[i].idle    - before->cores[i].idle;
        long long dw  = after->cores[i].iowait  - before->cores[i].iowait;
        long long dr  = after->cores[i].irq     - before->cores[i].irq;
        long long dsr = after->cores[i].softirq - before->cores[i].softirq;
        long long total = du + dn + ds + di + dw + dr + dsr;
        long long idle  = di + dw;
        util_out[i] = (total <= 0) ? 0.0
                    : 100.0 * (double)(total - idle) / (double)total;
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static uint32_t lcg(uint32_t *s) { *s = (*s) * 1664525u + 1013904223u; return *s; }
static float lcg_f(uint32_t *s) { return (float)(lcg(s) & 0xFFFFFF) / (float)0x1000000; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

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
 * 섹션 3b: [THREAD] 세팅 단계 스레드 워커
 *   노이즈 → 즉시병합 → 블러 → 침식 → 클램프 → BARRIER
 * ============================================================ */
static void *thread_worker(void *arg) {
    ThreadArg *a = (ThreadArg*)arg;
    int tid = a->thread_id;
    TerrainMap *tmap = a->tmap;
    pthread_barrier_t *bar = a->barrier;

    int chunk_start = (TOTAL_CHUNKS * tid) / g_num_threads;
    int chunk_end   = (TOTAL_CHUNKS * (tid + 1)) / g_num_threads;
    int row_start   = (MAP_HEIGHT * tid) / g_num_threads;
    int row_end     = (MAP_HEIGHT * (tid + 1)) / g_num_threads;

    /* 단계1: 노이즈 + 즉시 병합 */
    double t = now_ms();
    for (int ci = chunk_start; ci < chunk_end; ci++) {
        noise_compute_chunk(&tmap->chunks[ci], &tmap->params);
        ChunkDesc *c = &tmap->chunks[ci];
        for (int ly = 0; ly < CHUNK_SIZE; ly++) {
            int gy = c->pixel_y_start + ly;
            memcpy(&tmap->data[gy][c->pixel_x_start],
                   &c->height_data[ly * CHUNK_SIZE],
                   sizeof(float) * CHUNK_SIZE);
        }
    }
    a->elapsed_noise_ms = now_ms() - t;

        /* BARRIER 1: 세마포어 방식 */
    semaphore_barrier_wait(&sem_b1_mutex, &sem_b1_turnstile, &sem_b1_count, g_num_threads);

/* 단계2: baseline과 동일한 경계 블러
     *
     * 기존 병렬 코드의 adaptive row blur는 baseline의 terrain_blur()와
     * 연산 방식이 달라 결과가 달라질 수 있었다.
     * 따라서 블러 단계만 baseline과 동일하게 맞추기 위해
     * tid == 0 스레드 하나가 baseline terrain_blur(tmap)를 그대로 실행한다.
     */
    t = now_ms();
    if (tid == 0) {
        terrain_blur(tmap);
        a->elapsed_blur_ms = now_ms() - t;
    } else {
        a->elapsed_blur_ms = 0.0;
    }

        /* BARRIER 2: 세마포어 방식 */
    semaphore_barrier_wait(&sem_b2_mutex, &sem_b2_turnstile, &sem_b2_count, g_num_threads);

/* 단계3: 침식 (가상 구역 분할) */
    t = now_ms();
    float (*map)[MAP_WIDTH] = tmap->data;
    const float er = 0.004f, dr = 0.002f;
    int virtual_chunks = g_num_threads;
    int rows_per_virtual = MAP_HEIGHT / virtual_chunks;
    long drops_per_virtual = EROSION_DROPS / virtual_chunks;
    for (int v_id = 0; v_id < virtual_chunks; v_id++) {
        if (v_id % g_num_threads != tid) continue;
        int v_row_start = v_id * rows_per_virtual;
        int v_row_end   = (v_id == virtual_chunks - 1) ? MAP_HEIGHT : (v_id + 1) * rows_per_virtual;
        uint32_t rng = tmap->params.seed ^ 0xBAADF00D ^ (uint32_t)(v_id * 0x1111);
        for (long d = 0; d < drops_per_virtual; d++) {
            int fx = (int)(((rng = rng*1664525u+1013904223u)&0xFFFFFF)%(MAP_WIDTH-2))+1;
            int fy = (int)(((rng = rng*1664525u+1013904223u)&0xFFFFFF)%(v_row_end-v_row_start-2))+v_row_start+1;
            for (int step = 0; step < EROSION_STEPS; step++) {
            // 1칸씩 더 안쪽에서 빗방울을 소멸시키도록 안전 마진 강화
            // 기존: fy <= v_row_start || fy >= v_row_end - 1
            // 변경: fy <= v_row_start + 1 || fy >= v_row_end - 2
                if (fx < 1 || fx >= MAP_WIDTH - 1 || fy <= v_row_start + 1 || fy >= v_row_end - 2) {
                    break;
                }

                float hc = map[fy][fx], hm = hc; 
                int dx = 0, dy = 0;
            
                if (map[fy][fx+1] < hm) { hm = map[fy][fx+1]; dx = 1;  dy = 0;  }
                if (map[fy][fx-1] < hm) { hm = map[fy][fx-1]; dx = -1; dy = 0;  }
                if (map[fy+1][fx] < hm) { hm = map[fy+1][fx]; dx = 0;  dy = 1;  } 
                if (map[fy-1][fx] < hm) { hm = map[fy-1][fx]; dx = 0;  dy = -1; } 
            
                if (!dx && !dy) break;

                float diff = hc - hm;
                map[fy][fx] -= er * diff; 
                map[fy+dy][fx+dx] += dr * diff; // 이제 무조건 내 스레드 전용 공간 내에서만 안전하게 연산됨
            
                fx += dx; fy += dy;
        }
        }
    }
    /* 클램프 */
    for (int y = row_start; y < row_end; y++)
        for (int x = 0; x < MAP_WIDTH; x++)
            tmap->data[y][x] = clampf(tmap->data[y][x], HEIGHT_MIN, HEIGHT_MAX);
    a->elapsed_erosion_ms = now_ms() - t;

        /* BARRIER 3: 세마포어 방식 */
    semaphore_barrier_wait(&sem_b3_mutex, &sem_b3_turnstile, &sem_b3_count, g_num_threads);

/* ★ verify_barrier: 클램핑 완료 → PARENT 무결성 검사 허용 */
    pthread_barrier_wait(a->verify_barrier);

    a->elapsed_merge_ms = 0;

    /* ── 세팅 완료 후 플레이 단계 렌더 루프 진입 ──
     * play_barrier: PARENT가 CHILD 종료 + shared 초기화 완료할 때까지 대기
     * 그 후 render_thread_worker 루프와 동일하게 실행
     */
    pthread_barrier_wait(a->play_barrier);

    /* shared가 NULL이면 플레이 없이 종료 */
    if (!a->shared) return NULL;

    /* render_thread_worker 루프 재사용 */
    struct PlayShared *s = a->shared;
    int render_tid = a->thread_id;
    double elapsed = 0.0;

    int dc_start = render_tid * (DRAWCALL_COUNT / g_play_render_threads);
    int dc_end   = (render_tid == g_play_render_threads - 1)
                   ? DRAWCALL_COUNT
                   : dc_start + (DRAWCALL_COUNT / g_play_render_threads);

    while (1) {
        pthread_mutex_lock(&s->mutex);
        while (s->ready_buf < 0 && !s->finished)
            pthread_cond_wait(&s->cond_render, &s->mutex);
        if (s->finished && s->ready_buf < 0) {
            pthread_mutex_unlock(&s->mutex);
            break;
        }
        int buf_idx = s->ready_buf;
        int tick    = s->current_tick;
        pthread_mutex_unlock(&s->mutex);

        double t = now_ms();
        const DrawCall   *cmds      = s->buf[buf_idx];
        double           *frame_acc = s->frame_acc[buf_idx];
        const TerrainMap *tmap_r    = s->tmap;

        for (int pass = 0; pass < RENDER_PASSES; pass++) {
            for (int i = dc_start; i < dc_end; i++) {
                if (!(cmds[i].pass_mask & (1 << pass))) continue;
                float ray_ox = 512.0f, ray_oy = 2.0f, ray_oz = 512.0f;
                float ray_dx = cmds[i].pos_x - ray_ox;
                float ray_dy = cmds[i].pos_y - ray_oy;
                float ray_dz = cmds[i].pos_z - ray_oz;
                float ray_len = sqrtf(ray_dx*ray_dx + ray_dy*ray_dy + ray_dz*ray_dz);
                if (ray_len < 1e-6f) continue;
                ray_dx /= ray_len; ray_dy /= ray_len; ray_dz /= ray_len;
                float occlusion = 0.0f;
                for (int s2 = 0; s2 < RAYMARCH_STEPS; s2++) {
                    float tt = ray_len * (float)s2 / RAYMARCH_STEPS;
                    float sx = ray_ox + ray_dx * tt;
                    float sy = ray_oy + ray_dy * tt;
                    float sz = ray_oz + ray_dz * tt;
                    int mx = (int)clampf(sx, 0, MAP_WIDTH-1);
                    int mz = (int)clampf(sz, 0, MAP_HEIGHT-1);
                    if (sy < tmap_r->data[mz][mx]) occlusion += 1.0f;
                }
                occlusion /= RAYMARCH_STEPS;
                float pixel = (1.0f - occlusion)
                            * cmds[i].light_intensity
                            * (1.0f + sinf((float)tick * 0.1f + cmds[i].pos_x * 0.01f) * 0.05f);
                frame_acc[i] += (double)pixel;
            }
        }
        elapsed += now_ms() - t;

        pthread_mutex_lock(&s->mutex);
        s->done_count++;
        if (s->done_count == g_play_render_threads) {
            s->ready_buf  = -1;
            s->done_count = 0;
            pthread_cond_signal(&s->cond_physics);
        }
        pthread_mutex_unlock(&s->mutex);
    }

    s->render_ms_per_thread[render_tid] = elapsed;
    return NULL;
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
static double tick_render(const DrawCall *cmds, const TerrainMap *tmap,
                          double *frame_acc, int tick)
{
    double pixel_sum = 0.0;
    const float LX =  0.577f, LY = 0.577f, LZ = -0.577f;

    for (int pass = 0; pass < RENDER_PASSES; pass++) {
        for (int i = 0; i < DRAWCALL_COUNT; i++) {
            if (!(cmds[i].pass_mask & (1 << pass))) continue;

            float ray_ox = 512.0f, ray_oy = 2.0f, ray_oz = 512.0f;
    float ray_dx = cmds[i].pos_x - ray_ox;
    float ray_dy = cmds[i].pos_y - ray_oy;
    float ray_dz = cmds[i].pos_z - ray_oz;
    float ray_len = sqrtf(ray_dx*ray_dx + ray_dy*ray_dy + ray_dz*ray_dz);
    ray_dx /= ray_len; ray_dy /= ray_len; ray_dz /= ray_len;

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
                * (1.0f + sinf((float)tick * 0.1f + cmds[i].pos_x * 0.01f) * 0.05f);

            frame_acc[i] += (double)pixel;
            pixel_sum    += (double)pixel;
        }
    }
    return pixel_sum;
}

/* ============================================================
 * 섹션 9b: [PLAY][THREAD] 렌더 스레드 워커
 *   더블버퍼에서 명령서 받아 GPU렌더 실행
 * ============================================================ */
static void *render_thread_worker(void *arg) {
    RenderThreadArg *a = (RenderThreadArg*)arg;
    PlayShared *s = a->shared;
    int tid = a->thread_id;
    double elapsed = 0.0;

    /* 드로우콜 분할: DRAWCALL_COUNT / g_play_render_threads */
    int dc_start = tid * (DRAWCALL_COUNT / g_play_render_threads);
    int dc_end   = (tid == g_play_render_threads - 1)
                   ? DRAWCALL_COUNT
                   : dc_start + (DRAWCALL_COUNT / g_play_render_threads);

    while (1) {
        pthread_mutex_lock(&s->mutex);
        while (s->ready_buf < 0 && !s->finished)
            pthread_cond_wait(&s->cond_render, &s->mutex);
        if (s->finished && s->ready_buf < 0) {
            pthread_mutex_unlock(&s->mutex);
            break;
        }
        int buf_idx  = s->ready_buf;
        int tick     = s->current_tick;
        pthread_mutex_unlock(&s->mutex);

        /* 렌더링 (담당 드로우콜 범위만) */
        double t = now_ms();
        const DrawCall   *cmds      = s->buf[buf_idx];
        double           *frame_acc = s->frame_acc[buf_idx];
        const TerrainMap *tmap      = s->tmap;

        for (int pass = 0; pass < RENDER_PASSES; pass++) {
            for (int i = dc_start; i < dc_end; i++) {
                if (!(cmds[i].pass_mask & (1 << pass))) continue;
                float ray_ox = 512.0f, ray_oy = 2.0f, ray_oz = 512.0f;
                float ray_dx = cmds[i].pos_x - ray_ox;
                float ray_dy = cmds[i].pos_y - ray_oy;
                float ray_dz = cmds[i].pos_z - ray_oz;
                float ray_len = sqrtf(ray_dx*ray_dx + ray_dy*ray_dy + ray_dz*ray_dz);
                if (ray_len < 1e-6f) continue;
                ray_dx /= ray_len; ray_dy /= ray_len; ray_dz /= ray_len;
                float occlusion = 0.0f;
                for (int s2 = 0; s2 < RAYMARCH_STEPS; s2++) {
                    float tt = ray_len * (float)s2 / RAYMARCH_STEPS;
                    float sx = ray_ox + ray_dx * tt;
                    float sy = ray_oy + ray_dy * tt;
                    float sz = ray_oz + ray_dz * tt;
                    int mx = (int)clampf(sx, 0, MAP_WIDTH-1);
                    int mz = (int)clampf(sz, 0, MAP_HEIGHT-1);
                    if (sy < tmap->data[mz][mx]) occlusion += 1.0f;
                }
                occlusion /= RAYMARCH_STEPS;
                float pixel = (1.0f - occlusion)
                            * cmds[i].light_intensity
                            * (1.0f + sinf((float)tick * 0.1f + cmds[i].pos_x * 0.01f) * 0.05f);
                frame_acc[i] += (double)pixel;
            }
        }
        elapsed += now_ms() - t;

        /* 완료 통보 */
        pthread_mutex_lock(&s->mutex);
        s->done_count++;
        if (s->done_count == g_play_render_threads) {
            s->ready_buf  = -1;
            s->done_count = 0;
            pthread_cond_signal(&s->cond_physics);
        }
        pthread_mutex_unlock(&s->mutex);
    }

    s->render_ms_per_thread[tid] = elapsed;
    return NULL;
}

/* ============================================================
 * 섹션 11: 성능 출력
 * ============================================================ */
static void print_metrics(const PerfMetrics *m) {
    double cpu_total = m->cpu_user_ms + m->cpu_sys_ms;

    double par_total = m->thread_total_ms + m->parent_lighting_ms
                 + m->child1_ai_ms   + m->child2_particle_ms
                 + m->play_roundrobin_ms;
double P = (m->total_ms > 0) ? par_total / m->total_ms : 0;
if (P > 1.0) P = 1.0;
int effective_workers = (g_num_threads + 1 + g_child_count > g_play_render_threads + 1)
                      ? (g_num_threads + 1 + g_child_count)
                      : (g_play_render_threads + 1);
double speedup_theory = 1.0 / ((1.0-P) + P/(double)effective_workers);
double cpu_util = (m->total_ms > 0) ? cpu_total / m->total_ms * 100.0 : 0;

printf("\n");
printf("===========================================================\n");
printf("     게임 월드 병렬 실험 버전 (Thread×%d→%d + fork×%d)\n", g_num_threads, g_play_render_threads, g_child_count);
printf("===========================================================\n");
printf("[세팅 단계]\n");
printf("----- [THREAD×%d] 지형 맵 생성 -----\n", g_num_threads);
printf("  노이즈+병합:    %8.2f ms  <- BARRIER POINT 1\n", m->thread_noise_ms);
printf("  블러:           %8.2f ms  <- BARRIER POINT 2\n", m->thread_blur_ms);
printf("  침식:           %8.2f ms  <- BARRIER POINT 3\n", m->thread_erosion_ms);
printf("  히스토그램:     %8.2f ms\n", m->thread_integrity_ms);
printf("  소계:           %8.2f ms\n", m->thread_total_ms);
printf("  세팅 벽시계:    %8.2f ms\n", m->setting_wall_ms);
printf("-----------------------------------------------------------\n");
printf("  [PARENT] 조도 계산:     %8.2f ms  (Thread와 병렬)\n", m->parent_lighting_ms);
printf("  [CHILD1] 몬스터 AI:     %8.2f ms  (%s)\n",
       m->child1_ai_ms, (g_child_count >= 1) ? "fork 독립 실행" : "PARENT가 직접 수행");
printf("  [CHILD2] 파티클 4종:    %8.2f ms  (%s)\n",
       m->child2_particle_ms, (g_child_count >= 2) ? "fork 독립 실행" : "PARENT가 직접 수행");
printf("===========================================================\n");
printf("[플레이 단계] %d틱 | PARENT×1 + THREAD×%d\n", m->play_ticks, g_play_render_threads);
printf("  [PARENT] 물리/AI:   %8.2f ms  (틱평균 %.2f ms)\n",
       m->play_physics_ms, m->play_physics_ms / m->play_ticks);
printf("  [PARENT] 명령서:    %8.2f ms  (틱평균 %.2f ms)\n",
       m->play_cmd_ms, m->play_cmd_ms / m->play_ticks);
printf("  [THREAD] GPU렌더:   %8.2f ms  (스레드 최대 누적)\n", m->play_render_ms);
printf("  플레이 벽시계:      %8.2f ms\n", m->play_wall_ms);
printf("===========================================================\n");
printf("  총 소요 시간:          %8.2f ms\n", m->total_ms);
printf("===========================================================\n");
printf("[태스크별 비율]\n");
printf("  THREAD:       %5.1f%%   PARENT:      %5.1f%%\n",
       m->thread_total_ms/m->total_ms*100,
       m->parent_lighting_ms/m->total_ms*100);
printf("  CHILD1:       %5.1f%%   CHILD2:      %5.1f%%\n",
       m->child1_ai_ms/m->total_ms*100,
       m->child2_particle_ms/m->total_ms*100);
printf("  PLAY:         %5.1f%%\n",
       m->play_roundrobin_ms/m->total_ms*100);
printf("===========================================================\n");
printf("[CPU 성능 지표]\n");
printf("  CPU user 시간:  %8.2f ms\n", m->cpu_user_ms);
printf("  CPU sys  시간:  %8.2f ms\n", m->cpu_sys_ms);
printf("  CPU 활용률:     %8.1f%%   (= (user+sys) / wall × 100)\n", cpu_util);
printf("===========================================================\n");
printf("[세팅 단계 코어별 CPU 활용률]        [플레이 단계 코어별 CPU 활용률]\n");
for (int i = 0; i < m->num_cores; i++) {
    printf("  cpu%-2d: %5.1f%%                        cpu%-2d: %5.1f%%\n",
           i, m->setup_core_util[i],
           i, m->play_core_util[i]);
}
printf("===========================================================\n");
printf("[IPC pipe]\n");
printf("  pipe write:  %4d 회 | pipe read: %4d 회\n",
       m->ipc_pipe_write, m->ipc_pipe_read);
printf("  전송량:      %4d bytes\n", m->ipc_bytes);
printf("===========================================================\n");
printf("[결과 검증 — 병렬화 전후 동일해야 함]\n");
printf("  평균 조도:     %.6f\n", m->lighting_result);
printf("  AI 최고점수:   %.4f\n", m->ai_best_score);
printf("  파티클 에너지: %.6f\n", m->particle_final_energy);
printf("  무결성 실패:   %d\n", m->integrity_fails);
printf("===========================================================\n");
printf("[암달의 법칙 (%d worker 기준)]\n", effective_workers);
printf("  병렬 비율 P = %.1f%%\n", P*100);
printf("  이론 최대 Speedup = %.2fx\n", speedup_theory);
printf("===========================================================\n");

}
static void print_ascii_map(const TerrainMap *tmap) {
    printf("\n[ASCII 미니맵] 64x32:\n");
    const char *sh = " .:;+=xX$&#";
    for (int y=0;y<32;y++) {
        for (int x=0;x<64;x++) {
            float h=tmap->data[y*MAP_HEIGHT/32][x*MAP_WIDTH/64];
            int li=(int)(h*9); li=li<0?0:li>9?9:li;
            putchar(sh[li]);
        }
        putchar('\n');
    }
}


static void print_usage(const char *prog) {
    printf("사용법: %s [--threads N] [--children N] [--play-threads N]\n", prog);
    printf("  --threads N      세팅 단계 지형 생성 pthread 수 (1~%d, 기본 %d)\n",
           MAX_SETUP_THREADS, DEFAULT_NUM_THREADS);
    printf("  --children N     fork child 수 (0~2, 기본 %d). 남은 CHILD 작업은 PARENT가 수행\n", DEFAULT_CHILD_COUNT);
    printf("                   0=child 없음, 1=CHILD1(AI)만, 2=CHILD1+CHILD2\n");
    printf("  --play-threads N 플레이 단계 GPU 렌더 pthread 총 수 (1~%d, 기본 threads+%d)\n",
           MAX_PLAY_THREADS, DEFAULT_PLAY_EXTRA_THREADS);
    printf("예시:\n");
    printf("  %s --threads 2 --children 1\n", prog);
    printf("  %s --threads 4 --children 2 --play-threads 6\n", prog);
    printf("  %s --threads 8 --children 0 --play-threads 8\n", prog);
}

static int parse_int_arg(const char *s, int *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (*end != '\0') return 0;
    *out = (int)v;
    return 1;
}

static int parse_args(int argc, char **argv) {
    int play_threads_given = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 1;
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &g_num_threads)) {
                fprintf(stderr, "오류: --threads 뒤에는 정수가 필요합니다.\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--children") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &g_child_count)) {
                fprintf(stderr, "오류: --children 뒤에는 정수가 필요합니다.\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--play-threads") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &g_play_render_threads)) {
                fprintf(stderr, "오류: --play-threads 뒤에는 정수가 필요합니다.\n");
                return -1;
            }
            play_threads_given = 1;
        } else {
            fprintf(stderr, "알 수 없는 옵션: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }

    if (g_num_threads < 1 || g_num_threads > MAX_SETUP_THREADS) {
        fprintf(stderr, "오류: --threads는 1~%d 범위여야 합니다.\n", MAX_SETUP_THREADS);
        return -1;
    }
    if (g_child_count < 0 || g_child_count > 2) {
        fprintf(stderr, "오류: --children은 0~2 범위여야 합니다.\n");
        return -1;
    }
    if (!play_threads_given) {
        g_play_render_threads = g_num_threads + DEFAULT_PLAY_EXTRA_THREADS;
    }
    if (g_play_render_threads < g_num_threads) {
        fprintf(stderr, "오류: --play-threads는 --threads보다 작을 수 없습니다.\n");
        return -1;
    }
    if (g_play_render_threads < 1 || g_play_render_threads > MAX_PLAY_THREADS) {
        fprintf(stderr, "오류: --play-threads는 1~%d 범위여야 합니다.\n", MAX_PLAY_THREADS);
        return -1;
    }
    return 0;
}

/* ============================================================
 * 섹션 12: main — 병렬 오케스트레이션
 *
 *  [세팅] pthread×N + fork×K 동시 시작, 남은 CHILD 작업은 PARENT가 수행
 *  [플레이] waitpid 후 CHILD 종료 → pthread×2 추가 생성
 *           PARENT 틱루프 ↔ THREAD×6 GPU렌더 더블버퍼 병렬
 * ============================================================ */
int main(int argc, char **argv) {
    int arg_status = parse_args(argc, argv);
    if (arg_status > 0) return 0;
    if (arg_status < 0) return 1;

    int extra_play_threads = g_play_render_threads - g_num_threads;

    printf("=== 게임 월드 병렬 실험 버전 (Thread×%d→%d + fork×%d) ===\n",
           g_num_threads, g_play_render_threads, g_child_count);
    printf("시드: 0x%08X (고정) | 맵: %dx%d\n",
           FIXED_SEED, MAP_WIDTH, MAP_HEIGHT);
    printf("실험 설정: setup threads=%d, play render threads=%d, child processes=%d\n\n",
           g_num_threads, g_play_render_threads, g_child_count);

    TerrainMap *tmap = calloc(1, sizeof(TerrainMap));
    if (!tmap) { perror("calloc"); return 1; }
    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        int cx = i % CHUNKS_PER_ROW, cy = i / CHUNKS_PER_ROW;
        tmap->chunks[i].chunk_id = i;
        tmap->chunks[i].chunk_x  = cx; tmap->chunks[i].chunk_y = cy;
        tmap->chunks[i].pixel_x_start = cx * CHUNK_SIZE;
        tmap->chunks[i].pixel_y_start = cy * CHUNK_SIZE;
        tmap->chunks[i].height_data = malloc(sizeof(float) * CHUNK_SIZE * CHUNK_SIZE);
        if (!tmap->chunks[i].height_data) { perror("malloc"); return 1; }
    }

    PerfMetrics m = {0};
    double t0 = now_ms();

    CpuSnapshot snap_setup_before, snap_setup_after;
    CpuSnapshot snap_play_before,  snap_play_after;

    cpu_snapshot(&snap_setup_before);

    init_noise_params(&tmap->params);

    /* ── 커스텀 세마포어 초기화 ── */
    sem_init(&sem_b1_mutex, 0, 1);
    sem_init(&sem_b1_turnstile, 0, 0);
    sem_init(&sem_b2_mutex, 0, 1);
    sem_init(&sem_b2_turnstile, 0, 0);
    sem_init(&sem_b3_mutex, 0, 1);
    sem_init(&sem_b3_turnstile, 0, 0);

    /* ── barrier 초기화 (세팅 단계 4개 스레드용) ── */
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, g_num_threads);

    /* ── verify_barrier 초기화: 스레드 4개 + PARENT 1개 = 5개 ── */
    pthread_barrier_t verify_barrier;
    pthread_barrier_init(&verify_barrier, NULL, g_num_threads + 1);

    /* ── play_barrier 초기화: 세팅 스레드 + PARENT 1개 = 5개 ── */
    pthread_barrier_t play_barrier;
    pthread_barrier_init(&play_barrier, NULL, g_num_threads + 1);

    /* ── pthread 생성: setup thread 수는 실행 인자로 변경 가능 ── */
    pthread_t *tids = calloc((size_t)g_play_render_threads, sizeof(pthread_t));
    ThreadArg *targs = calloc((size_t)g_num_threads, sizeof(ThreadArg));
    if (!tids || !targs) { perror("calloc threads"); return 1; }
    double t_parallel = now_ms();

    for (int t = 0; t < g_num_threads; t++) {
        targs[t].thread_id   = t;
        targs[t].tmap        = tmap;
        targs[t].barrier      = &barrier;
        targs[t].play_barrier = &play_barrier;
        targs[t].verify_barrier = &verify_barrier;
        targs[t].shared       = NULL;  /* 나중에 설정 */
        targs[t].elapsed_noise_ms = targs[t].elapsed_blur_ms = 0;
        targs[t].elapsed_erosion_ms = targs[t].elapsed_merge_ms = 0;
        if (pthread_create(&tids[t], NULL, thread_worker, &targs[t])) {
            perror("pthread_create"); return 1;
        }
    }

    /* ── fork child 생성: g_child_count에 따라 0~2개 실행 ──
     * 공정한 비교를 위해 CHILD 작업량은 항상 동일하게 유지한다.
     *   --children 0 : fork 없음, PARENT가 CHILD1 + CHILD2를 모두 수행
     *   --children 1 : CHILD1은 fork, CHILD2는 PARENT가 수행
     *   --children 2 : CHILD1 + CHILD2 모두 fork
     */
    pid_t pids[2] = {-1, -1};
    int result_pipe[2][2];
    for (int c = 0; c < 2; c++) {
        if (pipe(result_pipe[c]) < 0) { perror("pipe"); return 1; }
    }

    if (g_child_count >= 1) {
        /* Child1: 몬스터 AI */
        pids[0] = fork();
        if (pids[0] < 0) { perror("fork child1"); return 1; }
        if (pids[0] == 0) {
            close(result_pipe[0][0]);
            close(result_pipe[1][0]); close(result_pipe[1][1]);
            double c1_t0 = now_ms();
            float result = child1_monster_ai(FIXED_SEED);
            double c1_elapsed = now_ms() - c1_t0;
            ssize_t w1 = write(result_pipe[0][1], &result,     sizeof(float)); (void)w1;
            ssize_t w2 = write(result_pipe[0][1], &c1_elapsed, sizeof(double)); (void)w2;
            close(result_pipe[0][1]);
            exit(0);
        }
    }

    if (g_child_count >= 2) {
        /* Child2: 파티클 시뮬레이션 */
        pids[1] = fork();
        if (pids[1] < 0) { perror("fork child2"); return 1; }
        if (pids[1] == 0) {
            close(result_pipe[1][0]);
            close(result_pipe[0][0]); close(result_pipe[0][1]);
            double c2_t0 = now_ms();
            float result = child2_particle_simulation(FIXED_SEED);
            double c2_elapsed = now_ms() - c2_t0;
            ssize_t w1 = write(result_pipe[1][1], &result,     sizeof(float)); (void)w1;
            ssize_t w2 = write(result_pipe[1][1], &c2_elapsed, sizeof(double)); (void)w2;
            close(result_pipe[1][1]);
            exit(0);
        }
    }

    /* parent는 사용하지 않는 write end를 닫는다. */
    close(result_pipe[0][1]);
    close(result_pipe[1][1]);

    /* ── PARENT: 조도 계산 (Thread/Child와 병렬) ── */
    double t_light = now_ms();
    m.lighting_result    = parent_lighting_simulation(FIXED_SEED);
    m.parent_lighting_ms = now_ms() - t_light;
    printf("[PARENT] 조도 완료: %.2f ms\n\n", m.parent_lighting_ms);

    /* ── fork되지 않은 CHILD 작업은 PARENT가 직접 수행한다. ── */
    float ai_result = 0.0f, particle_result = 0.0f;

    if (g_child_count == 0) {
        double c1_t0 = now_ms();
        ai_result = child1_monster_ai(FIXED_SEED);
        m.child1_ai_ms = now_ms() - c1_t0;
        m.ai_best_score = ai_result;
        printf("[PARENT->CHILD1] AI 결과: %.4f (%.2f ms)\n", ai_result, m.child1_ai_ms);
    }

    if (g_child_count < 2) {
        double c2_t0 = now_ms();
        particle_result = child2_particle_simulation(FIXED_SEED);
        m.child2_particle_ms = now_ms() - c2_t0;
        m.particle_final_energy = particle_result;
        printf("[PARENT->CHILD2] 파티클 결과: %.6f (%.2f ms)\n", particle_result, m.child2_particle_ms);
    }

    /* ── pipe read: fork로 실행한 child 결과만 읽는다. ── */
    if (g_child_count >= 1) {
        ssize_t r1 = read(result_pipe[0][0], &ai_result,      sizeof(float));  (void)r1;
        ssize_t r2 = read(result_pipe[0][0], &m.child1_ai_ms, sizeof(double)); (void)r2;
        m.ai_best_score = ai_result;
        printf("[CHILD1] AI 결과: %.4f (%.2f ms)\n", ai_result, m.child1_ai_ms);
    }

    if (g_child_count >= 2) {
        ssize_t r3 = read(result_pipe[1][0], &particle_result,      sizeof(float));  (void)r3;
        ssize_t r4 = read(result_pipe[1][0], &m.child2_particle_ms, sizeof(double)); (void)r4;
        m.particle_final_energy = particle_result;
        printf("[CHILD2] 파티클 결과: %.6f (%.2f ms)\n", particle_result, m.child2_particle_ms);
    }

    close(result_pipe[0][0]); close(result_pipe[1][0]);
    m.ipc_pipe_write = g_child_count * 2;
    m.ipc_pipe_read  = g_child_count * 2;
    m.ipc_bytes      = g_child_count * (int)(sizeof(float) + sizeof(double));

    /* ── waitpid: 실행한 child만 대기 ── */
    for (int c = 0; c < g_child_count; c++) {
        int status;
        waitpid(pids[c], &status, 0);
        printf("[Parent] Child %d 종료\n", c+1);
    }
    m.setting_wall_ms = now_ms() - t_parallel;
    pthread_barrier_destroy(&barrier);
    printf("\n[세팅 완료] 벽시계: %.2f ms\n\n", m.setting_wall_ms);

    /* ★ verify_barrier: 스레드 4개가 클램핑까지 완전히 마칠 때까지 대기 */
    pthread_barrier_wait(&verify_barrier);

    /* ── 커스텀 세마포어 리소스 정리 ── */
    sem_destroy(&sem_b1_mutex);
    sem_destroy(&sem_b1_turnstile);
    sem_destroy(&sem_b2_mutex);
    sem_destroy(&sem_b2_turnstile);
    sem_destroy(&sem_b3_mutex);
    sem_destroy(&sem_b3_turnstile);

    /* ================================================================
     * [blur 검증] blur_baseline.bin 과 픽셀 단위 diff
     *
     *   베이스라인이 저장한 blur 후 전체 맵과 비교.
     *   barrier 있는 버전: diff 픽셀 0개 (또는 float 오차 수준)
     *   barrier 없는 버전: race로 오염된 픽셀 수 + 최대 차이값 출력
     *
     *   DIFF_THRESH: 이 값 이상 차이나면 오염 픽셀로 판정.
     *   float 연산 오차(~1e-6)보다 크게 잡아야 false positive 방지.
     * ================================================================ */
    #define DIFF_THRESH 0.002f
    {
        FILE *bf = fopen("blur_baseline.bin", "rb");
        if (!bf) {
            printf("[검증] blur_baseline.bin 없음 — 베이스라인 먼저 실행하세요\n\n");
        } else {
            float *baseline = malloc(sizeof(float) * MAP_HEIGHT * MAP_WIDTH);
            if (!baseline) {
                printf("[검증] malloc 실패\n\n");
                fclose(bf);
            } else {
                size_t rd = fread(baseline, sizeof(float),
                                  (size_t)MAP_HEIGHT * MAP_WIDTH, bf);
                fclose(bf);

                if (rd != (size_t)MAP_HEIGHT * MAP_WIDTH) {
                    printf("[검증] 파일 크기 불일치 (읽은 floats: %zu / 예상: %d)\n\n",
                           rd, MAP_HEIGHT * MAP_WIDTH);
                } else {
                    int diff_count = 0;
                    float max_diff = 0.0f;
                    int max_y = 0, max_x = 0;

                    for (int y = 0; y < MAP_HEIGHT; y++) {
                        for (int x = 0; x < MAP_WIDTH; x++) {
                            float d = fabsf(tmap->data[y][x]
                                          - baseline[y * MAP_WIDTH + x]);
                            if (d > DIFF_THRESH) {
                                diff_count++;
                                if (d > max_diff) {
                                    max_diff = d;
                                    max_y = y; max_x = x;
                                }
                            }
                        }
                    }

                    printf("\n╔══════════════════════════════════════════════════════════╗\n");
                    printf("║          [blur 검증] vs blur_baseline.bin               ║\n");
                    printf("╠══════════════════════════════════════════════════════════╣\n");
                    printf("║ 오염 픽셀 수:  %8d / %d                        ║\n",
                           diff_count, MAP_HEIGHT * MAP_WIDTH);
                    printf("║ 최대 차이값:   %8.6f  at [%d][%d]               ║\n",
                           max_diff, max_y, max_x);
                    if (diff_count == 0)
                        printf("║ 결과: ✓ PASS — barrier 동기화 정상                    ║\n");
                    else
                        printf("║ 결과: ✗ FAIL — Race Condition 감지 (%d픽셀 오염)     ║\n",
                               diff_count);
                    printf("╚══════════════════════════════════════════════════════════╝\n\n");
                }
                free(baseline);
            }
        }
    }

    /* 히스토그램 (스레드 전체 클램핑 완료 후 안전) */
    double t_post = now_ms();
    terrain_histogram(tmap);
    m.thread_merge_ms = now_ms() - t_post;

    pthread_barrier_destroy(&verify_barrier);

    cpu_snapshot(&snap_setup_after);
    cpu_calc_util(&snap_setup_before, &snap_setup_after,
                  m.setup_core_util, &m.num_cores);

    /* ================================================================
     * [플레이 단계]     *   세팅 스레드는 play_barrier 대기 중 → shared 설정 후 해제
     *   추가 스레드 생성 (id 4, 5)
     *   PARENT가 play_barrier 통과 → 스레드 6개 렌더 루프 시작
     * ================================================================ */
    cpu_snapshot(&snap_play_before);

    printf("[PLAY] 세팅 스레드 재사용 + 추가 스레드 %d개 생성 → 총 %d개\n",
           extra_play_threads, g_play_render_threads);

    PlayShared shared = {0};
    shared.buf[0]       = malloc(sizeof(DrawCall) * DRAWCALL_COUNT);
    shared.buf[1]       = malloc(sizeof(DrawCall) * DRAWCALL_COUNT);
    shared.frame_acc[0] = calloc(DRAWCALL_COUNT, sizeof(double));
    shared.frame_acc[1] = calloc(DRAWCALL_COUNT, sizeof(double));
    shared.tmap         = tmap;
    shared.ready_buf    = -1;
    shared.done_count   = 0;
    shared.finished     = 0;
    shared.total_ticks  = PLAY_TICKS;
    shared.current_tick = 0;
    shared.pixel_grand  = 0.0;
    memset(shared.render_ms_per_thread, 0, sizeof(shared.render_ms_per_thread));
    pthread_mutex_init(&shared.mutex, NULL);
    pthread_cond_init(&shared.cond_render, NULL);
    pthread_cond_init(&shared.cond_physics, NULL);

    /* 세팅 스레드에 shared 포인터 설정 */
    for (int t = 0; t < g_num_threads; t++)
        targs[t].shared = &shared;

    /* 추가 렌더 스레드 생성 */
    RenderThreadArg *extra_args = NULL;
    if (extra_play_threads > 0) {
        extra_args = calloc((size_t)extra_play_threads, sizeof(RenderThreadArg));
        if (!extra_args) { perror("calloc extra_args"); return 1; }
        for (int i = 0; i < extra_play_threads; i++) {
            extra_args[i].thread_id = g_num_threads + i;
            extra_args[i].shared    = &shared;
            pthread_create(&tids[g_num_threads + i], NULL,
                           render_thread_worker, &extra_args[i]);
        }
    }

    /* PARENT가 play_barrier 통과 → 세팅 스레드 렌더 루프 진입 */
    pthread_barrier_wait(&play_barrier);

    /* ── PARENT: 틱 루프 ── */
    uint32_t rng = FIXED_SEED ^ 0xCAFEF00D;
    double t_physics = 0.0, t_cmd = 0.0;
    double t_play = now_ms();

    for (int tick = 0; tick < PLAY_TICKS; tick++) {
        int write_buf = tick & 1;

        double t = now_ms();
        int hits = tick_physics_ai(tmap, &rng, tick);
        t_physics += now_ms() - t;

        t = now_ms();
        tick_build_commands(tmap, shared.buf[write_buf], hits, &rng);
        t_cmd += now_ms() - t;

        if (tick > 0) {
            pthread_mutex_lock(&shared.mutex);
            while (shared.ready_buf >= 0)
                pthread_cond_wait(&shared.cond_physics, &shared.mutex);
            pthread_mutex_unlock(&shared.mutex);
        }

        pthread_mutex_lock(&shared.mutex);
        shared.ready_buf    = write_buf;
        shared.current_tick = tick;
        pthread_cond_broadcast(&shared.cond_render);
        pthread_mutex_unlock(&shared.mutex);

        if ((tick+1) % (PLAY_TICKS/10) == 0)
            printf("[PLAY] 틱 %3d/%d | 물리: %.1fms | 명령서: %.1fms (누적)\n",
                   tick+1, PLAY_TICKS, t_physics, t_cmd);
    }

    /* 마지막 틱 렌더 완료 대기 */
    pthread_mutex_lock(&shared.mutex);
    while (shared.ready_buf >= 0)
        pthread_cond_wait(&shared.cond_physics, &shared.mutex);
    shared.finished = 1;
    pthread_cond_broadcast(&shared.cond_render);
    pthread_mutex_unlock(&shared.mutex);

    /* ── 전체 스레드 조인 (세팅4 + 추가2) ── */
    for (int t = 0; t < g_play_render_threads; t++) pthread_join(tids[t], NULL);

    cpu_snapshot(&snap_play_after);
    cpu_calc_util(&snap_play_before, &snap_play_after,
                  m.play_core_util, &m.num_cores);

    double play_wall = now_ms() - t_play;

    /* 세팅 스레드 elapsed 수집 */
    double max_noise = 0, max_blur = 0, max_erosion = 0, max_render = 0;
    for (int t = 0; t < g_num_threads; t++) {
        if (targs[t].elapsed_noise_ms   > max_noise)   max_noise   = targs[t].elapsed_noise_ms;
        if (targs[t].elapsed_blur_ms    > max_blur)    max_blur    = targs[t].elapsed_blur_ms;
        if (targs[t].elapsed_erosion_ms > max_erosion) max_erosion = targs[t].elapsed_erosion_ms;
    }
    for (int t = 0; t < g_play_render_threads; t++)
        if (shared.render_ms_per_thread[t] > max_render)
            max_render = shared.render_ms_per_thread[t];

    m.thread_noise_ms   = max_noise;
    m.thread_blur_ms    = max_blur;
    m.thread_erosion_ms = max_erosion;
    m.thread_total_ms   = max_noise + max_blur + max_erosion
                        + m.thread_merge_ms + m.thread_integrity_ms;

    printf("[PLAY] 완료\n");
    printf("[PLAY] 물리/AI:  %8.2f ms (틱평균 %.2f ms)\n",
           t_physics, t_physics / PLAY_TICKS);
    printf("[PLAY] 명령서:   %8.2f ms (틱평균 %.2f ms)\n",
           t_cmd, t_cmd / PLAY_TICKS);
    printf("[PLAY] GPU렌더:  %8.2f ms (스레드 최대 누적)\n", max_render);
    printf("[PLAY] 병렬 벽시계: %.2f ms\n\n", play_wall);

    m.play_ticks         = PLAY_TICKS;
    m.play_physics_ms    = t_physics;
    m.play_cmd_ms        = t_cmd;
    m.play_render_ms     = max_render;
    m.play_wall_ms       = play_wall;
    m.play_roundrobin_ms = play_wall;

    pthread_mutex_destroy(&shared.mutex);
    pthread_cond_destroy(&shared.cond_render);
    pthread_cond_destroy(&shared.cond_physics);
    pthread_barrier_destroy(&play_barrier);
    free(shared.buf[0]); free(shared.buf[1]);
    free(shared.frame_acc[0]); free(shared.frame_acc[1]);

    m.total_ms = now_ms() - t0;

    struct rusage ru_s, ru_c;
    getrusage(RUSAGE_SELF,     &ru_s);
    getrusage(RUSAGE_CHILDREN, &ru_c);
    m.cpu_user_ms = (ru_s.ru_utime.tv_sec  + ru_c.ru_utime.tv_sec)  * 1e3
                  + (ru_s.ru_utime.tv_usec + ru_c.ru_utime.tv_usec) / 1e3;
    m.cpu_sys_ms  = (ru_s.ru_stime.tv_sec  + ru_c.ru_stime.tv_sec)  * 1e3
                  + (ru_s.ru_stime.tv_usec + ru_c.ru_stime.tv_usec) / 1e3;

    print_ascii_map(tmap);
    print_metrics(&m);

    free(extra_args);
    free(tids);
    free(targs);
    for (int i = 0; i < TOTAL_CHUNKS; i++) free(tmap->chunks[i].height_data);
    free(tmap);
    return 0;
}