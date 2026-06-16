# Analysis 4 — Prediction-Timing Reframing (Final Synthesis): 작업일지 (Worklog)

**담당**: Donghyun · **작성**: 2026-06-17, 발표 D-1
**전제**: `Background.md`(진단·근거) · `plan.md`(실행계획)
**스크립트**: `Analysis4_final/final_analysis.ipynb`
**산출 위치**: `Analysis4_final/out/`

> **구조**: 1급 단위는 **시점 사다리(Timing Ladder) T0→T3**. Prep/Decomposition/Drift/Visualization/Validation은 사다리를 떠받치는 부속 단계. 아래는 단계별 결정·발견·보정 기록. 최종 종합은 `result.md`.

---

## Prep — 3원 머지 & 검증 ✅

- **작업**: base(`features_partB_v2.csv`, n=6249) + channel(`step1_channel_features.csv`) + post-upload(`cleaned_USvideos.csv`에서 `trending_lag`/`days_on_trending`/`log_likes`/`log_comment_count`/`comments_disabled`/`ratings_disabled` 발췌) → `video_id` inner merge.
- **결과**: 두 머지 모두 `assert len==len(base)` 통과 → **행수 6249 불변**. shape=(6249, 28). `comments_disabled`/`ratings_disabled` bool→int, `hour_bin` ordered Categorical(`["00-05","06-11","12-17","18-23"]`) 지정.
- **판단**: post-upload 컬럼 결측 0(inner=left 동일 = 정본 정합성 확인). T0~T3 모든 피처 확보.
- **산출**: `out/timing_merged.csv`.

## Timing Ladder — 4단 시점 사다리 T0→T3 ✅ (핵심)

- **작업**: A3 step3 `perf()`/category 필터 구조를 미러한 `evaluate_model()`로 T0~T3 4단을 rand split(rs=42)·LR/RF/GBM 평가. LR은 OLS formula(`C(hour_bin, Treatment('18-23')) + C(publish_weekday) + C(category)`), RF/GBM은 get_dummies 행렬.
- **결과 (rand test R²)**:

  | rung | LR | RF | GBM |
  |---|---|---|---|
  | T0 콘텐츠 | **0.1188** | 0.1115 | 0.1274 |
  | T1 +채널 | **0.2956** | 0.3036 | 0.3241 |
  | T2 +트렌딩 | 0.4281 | 0.5170 | 0.5121 |
  | T3 +참여도 | 0.8408 | 0.8577 | 0.8579 |

- **충실성 검증 통과**: T0 LR=0.11879 ≈ A3 0.1188, T1 LR=0.29560 ≈ A3 0.2956 (소수점 일치, `assert <1e-3` 통과). → base/formula/split 재구성 정확.
- **단조 상승 확인**: LR·RF·GBM **3모델 전부** T0<T1<T2<T3. "정확도 = 예측 시점의 함수" 수치 성립.
- **산출**: `out/timing_ladder_r2.csv` (rand 3모델 + `LR_time_R2`).

## Decomposition — 잔여분산 분해 ✅

- **작업**: T1(업로드시점 천장) 기준 `1−R²_T1`을 분해. 회수분=`R²_T3−R²_T1`, 잔존분=`1−R²_T3`.
- **결과**:
  - T1 설명분 = **0.2956 (29.6%)**
  - 사후정보 회수분(T3−T1) = **0.5452 (54.5%)**
  - 구조적 잔존분(1−T3) = **0.1592 (15.9%)**
- **⚠️ 중요 발견 — A3 프레이밍 수정 필요**: A3 evaluation §5는 "잔여 70%는 *대부분* 구조적 한계(sparsity·survivorship·관측불가)"로 추정했으나, **실측은 70.4% 중 54.5%p가 사후정보로 회수되고 진짜 구조적 잔존은 15.9%p뿐**. → 오류가 아니라 A4 논지 **강화**: "잔여는 노이즈가 아니라 *사후에만 보이는 정보*다. 시점만 옮기면 절반 이상 돌아온다." result.md에서 A3 표현을 이 수치에 맞춰 조정.
- **산출**: `out/timing_variance_decomp.csv`.

## Drift — 시간 일반화 진단 & RF 0.245 재라벨링 ✅

- **time-split 정렬 버그 수정 후 재실행**: 최초 실행은 `evaluate_model` time 분기가 `publish_time` 정렬 없이 행 순서대로 잘라(데이터가 views 내림차순) **고조회수 train / 저조회수 test** 분할이 됨 → time R²이 −11.38/−7.67로 터지고 A3(−1.27/−0.57)과 모순. cut 전 `sort_values("publish_time")` 추가로 수정(죽은 `time_split()` 헬퍼도 제거).
- **수정 후 결과 (LR time-split R²)**:

  | rung | time R² | 비고 |
  |---|---|---|
  | T0 | **−1.2711** | A3 정확 재현 |
  | T1 | **−0.5680** | A3 정확 재현 |
  | T2 | −0.1438 | 신규 |
  | T3 | **+0.7170** | 신규 — 양수 전환 |

- **발견 — drift도 시점의 함수**: time R²이 −1.27 → −0.57 → −0.14 → **+0.72로 단조 회복**. 예측 시점을 뒤로 옮길수록 temporal drift가 풀린다 → "drift는 별개 결함이 아니라 *시점 의존성의 한 증상*"(Background §5-4)이 수치로 입증. T3(완전 사후)에서야 시간 일반화가 양수.
- **RF R²=0.245 재라벨링**: Part C RF 헤드라인은 `trending_lag` 포함값 → 사다리상 **T2(트렌딩 직후) 예측**에 해당. "누수"가 아니라 "다른 예측 시점". (참고: A4 T2 rand RF=0.517로 Part C 0.245보다 높은데, A4 T2가 채널 피처까지 포함하기 때문 — Part C는 채널 미사용.)
- **산출**: `out/timing_ladder_r2.csv`의 `LR_time_R2` 컬럼.

## Visualization — 사다리 곡선 & 분해 ✅

- **작업**: (1) 사다리 곡선 — x=시점(T0→T3), y=R²(LR/RF/GBM 3선), T0/T1=「Pre-Upload」·T2/T3=「Post-Upload」 음영. (2) 70% 분해 누적막대(29.6/54.5/15.9). 팔레트 PRIMARY `#3B4FE4`/ACCENT `#1A7F5A`/GRAY `#C8CDD6`.
- **보정**: 폰트 `DejaVu Sans` 사용(라벨 전부 영문 → 글리프 깨짐 없음, Malgun Gothic 불요).
- **산출**: `out/fig_timing_ladder.png`, `out/fig_timing_variance.png`.

## Validation — T0/T1 재현 검증 ✅

- T0 LR=0.1188, T1 LR=0.2956 자동 검증(`assert <1e-3`) 통과 + 단조성 플래그 통과.

---

## 보정·이슈 메모

- **파일 네이밍 정리**: A4 산출물을 step 접두사에서 **`timing_*`/`fig_timing_*`**로 변경(시점 분석임을 드러냄). 스크립트는 `final_analysis.ipynb`(종합 위상) 유지. `.py` 사본은 제거되어 노트북이 정본.
- **time-split 정렬 버그(해결)**: 위 Drift. 헤드라인 사다리(rand)·분산 분해·그림은 time-split과 무관해 영향 없었고, `LR_time_R2` 컬럼만 재생성으로 교정.
- **⚠️ T3 회수분의 성격 — 정직성 핵심**: T3 참여도(`log_likes`/`log_comment_count`)는 조회수와 **기계적으로 동반 변동**(같은 시청자가 보고 누름)한다. 즉 T3 R²=0.84는 "예측을 잘했다"기보다 **"답을 이미 포함한 정보를 넣었다"**에 가깝다. → 회수분 54.5%p를 "예측 성과"로 읽으면 안 됨. 오히려 *"업로드시점 천장(0.30)을 넘으려면 답과 동반하는 사후정보를 써야 한다"* = 시점 논제의 가장 강한 형태. result.md에 반드시 명시.
- **헤드라인 고정**: 업로드시점 정직 예측치는 **T1 = 0.296(OOF)**. T2/T3은 "사후=다른 시점, 시연" 라벨 강제.

---

**다음 문서**: `result.md` — 최종 종합·해석·전체 프로젝트 결론
