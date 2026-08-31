# 2026 OS 과제 코드

프로세스/스레드 병렬화 및 동기화 기법 실험 코드 모음 (game world 시뮬레이션 기반).

## 파일 구성

- `final_baseline.c` — 직렬(baseline) 버전
- `final_parallel_experiment.c` — 병렬화 실험
- `game_world_parallel_mutex_barrier.c` — mutex + barrier 기반 동기화
- `game_world_parallel_semaphore_barrier.c` — semaphore + barrier 기반 동기화
- `game_world_parallel_monitor_barrier.c` — monitor + barrier 기반 동기화
- `da_child2.c`, `da_child4.c`, `da_child6.c` — 자식 프로세스(fork) 개수별 실험 (2/4/6)
- `da_thread.c`, `da_thread_thread6.c` — 스레드 기반 병렬화 실험
- `na_thread.c`, `na_thread_opt.c` — 스레드 기반 실험 및 최적화 버전
- `ga_child.c` — 자식 프로세스 관련 실험
- `syncronize.c` — 동기화 실험 코드

## 문서

- `공통기준.md` — 실험 공통 기준
- `동기화.md` — 동기화 기법 정리
- `병렬.md` — 병렬화 관련 정리
- `병렬측정지표가이드.md.docx` — 병렬 성능 측정 지표 가이드
