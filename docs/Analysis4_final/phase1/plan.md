# Analysis 4 — Prediction-Timing Reframing: 상세 작업계획서 (Execution Plan)

**담당**: Donghyun (Part B 확장 — 콘텐츠/채널 트랙 → 예측 시점 재프레이밍)
**작성**: 2026-06-17, 발표 D-1
**전제 문서**: `docs/Analysis4_final/Background.md` (노선 진단·근거) · `docs/Analysis3_channel/phase1/evaluation.md` (70% 분해)
**입력** (3원 머지, `video_id` 기준):
- **base** `Analysis1_title/LR/out/features_partB_v2.csv` (T0 콘텐츠 피처 + `hour_bin`/`publish_weekday`/`publish_time`/`category`/`log_views` — **A3가 쓴 정본 base, T0/T1 재현 보장**)
- **channel** `Analysis3_channel/out/step1_channel_features.csv` (`chan_mean_oof`/`chan_mean_naive`/`chan_freq` — A3 산출)
- **post-upload** `dataset/cleaned_USvideos.csv` (T2/T3 사후변수 `trending_lag`/`days_on_trending`/`log_likes`/`log_comment_count`/`comments_disabled`/`ratings_disabled`만 발췌 머지)
**작업 위치**: `Analysis4_final/` (스크립트 + 산출 CSV/PNG)
**하니스**: 사다리 *개념*은 `Analysis1_title/LR/r2_ceiling_test.py`, **평가 하니스는 A3 `step3_channel_analysis.ipynb`의 `perf()` 미러** (단일 rand split; ⚠️ 아래 §B 참조)
**환경**: Anaconda `dm`, Python 3.12, VS Code

> **용어**: "Analysis"는 프로젝트 전체 단위(제목=A1, 태그=A2, 채널=A3, 재프레이밍=A4). A4의 1급 단위는 **시점 사다리(Timing Ladder) T0~T3**이며, Prep/Decomposition/Drift/Visualization/Documentation은 사다리를 떠받치는 부속 단계. "Rung(시점)"은 사다리의 단(T0~T3).
> **핵심 방법론 포인트**: 누수 경계 = **예측 시점**. `trending_lag`/참여도는 "버그"가 아니라 "더 나중 시점에 관측되는 정보". A4는 각 시점에서 R²가 어떻게 변하는지를 사다리로 보인다. 이 재해석이 A4의 차별화 지점.

---

## A. 예측 시점 정의 (누수 경계의 재프레이밍)

A1~A3는 "업로드시점 변수만"이 비협상이었다. **A4는 그 경계를 의도적으로 넘되, 넘는 것 자체를 "예측 시점 이동"으로 명시**한다.

| 시점(rung) | 추가 정보 | 관측 가능 시점 | 누수? |
|---|---|---|---|
| **T0 콘텐츠** | 제목·태그·시간·카테고리 | 업로드 직전 | ✗ (업로드시점) |
| **T1 +채널** | `chan_mean_oof`, `chan_freq_log` | 업로드 직전 (과거실적) | ✗ (업로드시점, OOF로 same-video 차단) |
| **T2 +트렌딩 메타** | `trending_lag`, `days_on_trending` | **트렌딩 진입 후** | △ (의도된 시점이동) |
| **T3 +참여도** | `log_likes`, `log_comment_count`, `comments_disabled`, `ratings_disabled` | **참여 발생 후** | △ (의도된 시점이동) |

- **헤드라인 예측치 = T1 (업로드시점 천장, OOF rand R²≈0.30).** T2/T3은 "성과"가 아니라 **"더 나중에 보면 당연히 더 잘 보인다"의 시연**(Background §5-1).
- T2/T3은 보고 시 반드시 **"사후정보 = 다른 예측 시점"** 라벨 부착. 업로드시점 예측 성과로 혼동 금지.

---

## B. 확정된 설계 결정 (A3 step3 코드로 검증됨)

| # | 항목 | 확정값 | 근거 |
|---|---|---|---|
| 1 | 타깃 | `log_views` | 전 분석 일관 |
| 2 | **T0** formula | `log_views ~ title_len + caps_ratio + exclaim_cnt + question_cnt + has_number + has_bracket + tag_cnt + C(hour_bin, Treatment(reference='18-23')) + C(publish_weekday) + C(category)` | **A3 `F_content` 그대로** → rand test R²=0.1188 재현 |
| 3 | **T1** formula | T0 + `chan_mean_oof` + `chan_freq_log` (`= log1p(chan_freq)`) | **A3 `F_chan` 그대로** → rand test R²=0.2956 재현 |
| 4 | **T2** formula | T1 + `trending_lag` + `days_on_trending` (수치형) | 트렌딩 시점 정보 |
| 5 | **T3** formula | T2 + `log_likes` + `log_comment_count` + `comments_disabled` + `ratings_disabled` (bool→int) | 참여 시점 정보 |
| 6 | `hour_bin` | `pd.Categorical(["00-05","06-11","12-17","18-23"])` 순서 지정 | A3 동일 (양 파일에 컬럼 존재, 생성 불요) |
| 7 | rand split | `train_test_split(df.dropna(subset=["publish_time"]), test_size=0.2, random_state=42)` | A3 동일 |
| 8 | **test category 필터** | `te = te[te["category"].isin(set(tr["category"].unique()))]` | A3 `perf()` 동일 — 재현에 필수 |
| 9 | 평가 split | 단일 rand split(rs=42) test R²로 rung 간 비교 (A3 단일 split 0.2956 재현) — **CV 미적용** (사다리는 rung 동등 비교가 목적, 단일 split로 충분) | A3 `perf()` 미러 |
| 10 | 모델 | LR(statsmodels OLS, **재현 앵커**) + RF/GBM(보조선, 단조성 확인) | LR이 0.119/0.296 앵커, RF/GBM은 사다리 일관성용 |
| 11 | 헤드라인 | **T1(OOF)** | 업로드시점 정직 천장; T2/T3은 시연 |
| 12 | dedup | base가 이미 dedup됨(n=6249). 머지 후 행수 불변 assert | A3 `assert len(df)==len(base)` 미러 |

> **⚠️ 하니스 주의 (재현 핵심)**: `r2_ceiling_test.py`의 eval 함수는 **재사용 금지**. 그것은 `cleaned_USvideos.csv`를 base로 `publish_hour`/`publish_weekday`를 **수치형**으로 넣고, hour_bin Treatment·category 필터가 없어 **T0=0.119를 재현 못 한다.** A4는 **A3 step3의 `perf()` + formula 방식을 그대로 미러**하고(단일 rand split) `FEATURE_SETS`만 T0~T3 4단으로 확장한다. (r2_ceiling_test는 "사후변수 단계 추가" 아이디어의 출처일 뿐.)
> **RF/GBM 주의**: RF/GBM은 `ColumnTransformer`(category 원핫 + 나머지 passthrough)로 별도 구성. A3 step4 RF 수치(0.304 등)와 정확히 일치하진 않을 수 있음 — 사다리에선 **rung 간 단조성**이 목적이지 A3 재현이 목적 아님. LR선이 재현 앵커.

---

## Prep — 셋업 & 3원 머지·검증

- **작업**
  - `Analysis4_final/` 스크립트 셋업. imports: `numpy, pandas, statsmodels.formula.api as smf, sklearn.{metrics,model_selection,ensemble,preprocessing,compose,pipeline}, matplotlib`
  - **base** `features_partB_v2.csv`(`parse_dates=["publish_time"]`) 로드
  - **channel** `step1_channel_features.csv`에서 `[video_id, chan_mean_oof, chan_mean_naive, chan_freq]` → `merge(on="video_id", how="inner")`, `assert len==len(base)`, `chan_freq_log=log1p(chan_freq)`
  - **post-upload** `cleaned_USvideos.csv`에서 `[video_id, trending_lag, days_on_trending, log_likes, log_comment_count, comments_disabled, ratings_disabled]` → `merge(on="video_id", how="left")`
  - `comments_disabled`/`ratings_disabled` bool→int. `hour_bin` ordered Categorical 지정.
  - **검증**: 3원 머지 후 행수=6249 불변, T2/T3 컬럼 결측 0 확인
- **산출**: `Analysis4_final/out/timing_merged.csv`
- **완료 기준**: 4단 피처 결측 없이 확보, 행수 6249 유지

## Timing Ladder — 4단 시점 사다리 T0→T3 (핵심)

- **작업**
  - A3 `perf(F, tr, te)` 구조 복제 (category 필터 포함). CV는 미적용 — 단일 rand split로 rung 비교.
  - `FEATURE_SETS = {T0:F_content, T1:F_chan, T2:F_t2, T3:F_t3}` (B표 #2~5)
  - 동일 rand split(rs=42)으로 각 rung LR 평가 + (보조) RF/GBM은 `ColumnTransformer` 파이프라인
  - **충실성 체크 (필수)**: T0 rand test R²≈0.1188, T1≈0.2956 매칭 → 불일치 시 base/formula/split 점검 후 진행
- **산출**: `Analysis4_final/out/timing_ladder_r2.csv`
  컬럼: `rung, features_desc, observable_at, LR_R2, RF_R2, GBM_R2`
- **완료 기준**: T0<T1<T2<T3 **단조 상승** 확인 + T0/T1 재현 검증 통과

## Decomposition — 잔여분산 분해 (70%의 정직한 해부)

- **작업**
  - T1 잔여분산 `1 − R²_T1 ≈ 0.704`(A3 step3 실측) 기준:
    - **사후정보 회수분** = `R²_T3 − R²_T1` (시점 이동으로 풀리는 분산 = (3b))
    - **끝내 잔존분** = `1 − R²_T3` (T3까지 가도 안 풀림 = 구조적 한계 (1)sparsity+(2)survivorship+(3a)관측불가)
  - 표로 "0.704 = 회수분 + 잔존분" 제시. 잔존분은 Background §1-2 (1)(2)(3a)로 정성 귀속(정량 분리는 범위 외 — 한계 명시)
- **산출**: `Analysis4_final/out/timing_variance_decomp.csv` (rung별 explained/recovered/residual)
- **완료 기준**: "T1 천장 0.30 → T3까지 회수 X% → 끝내 잔존 Y%" 수치 보고

## Drift — temporal drift & RF R²=0.245 재라벨링 (권장)

- **작업**
  - 각 rung을 A3 `time_split()`으로도 평가 → time R²이 음수에서 어떻게 변하는지 보고(drift = 시점 의존성의 한 증상, Background §5-4). A3 실측: T0 time R²=−1.27, T1=−0.57.
  - **A3 모순 #4 해소 시연**: Part C RF 헤드라인 R²=0.245가 `trending_lag` 포함값 → 사다리상 **T2(트렌딩 직후) 예측**에 해당함을 명시. "누수"가 아니라 "다른 시점".
- **산출**: `timing_ladder_r2.csv`에 `LR_time_R2` 컬럼 추가 + 재라벨링 메모(worklog)
- **완료 기준**: time-split 곡선 보고 + RF 0.245의 시점 귀속 명시
- **주의**: temporal drift 근본 해결 아님(데이터 한계) → 향후 과제 명시

## Visualization — 시각화 (발표용)

- **작업**
  - **사다리 곡선**: x=예측 시점(T0→T3), y=R²(LR 앵커선 + RF/GBM 보조선) — 단조 상승. T0/T1=「업로드시점」, T2/T3=「사후시점」 음영 구분 + "trending_lag = 트렌딩 시점 정보" 주석
  - **70% 분해 누적막대**: T1 설명분 / 사후정보 회수분 / 끝내 잔존분
  - 스타일: A1~A3 팔레트(PRIMARY `#3B4FE4`, ACCENT `#1A7F5A`, GRAY `#C8CDD6`), 한글폰트 Malgun Gothic, dpi=200
- **산출**: `Analysis4_final/out/fig_timing_ladder.png` (+ `fig_timing_variance.png`)
- **완료 기준**: "정확도 = 예측 시점의 함수"를 한 그림으로 보여주는 시각화 ≥1

## Documentation — 문서화 & 발표 반영

- **작업**
  - `docs/Analysis4_final/phase1/worklog.md`: 진행 중 결정·발견(T0/T1 재현값, 실제 회수분/잔존분, drift 곡선, RF 0.245 재라벨링)
  - `docs/Analysis4_final/phase1/result.md`: 최종 종합(A3 패턴)
  - 발표 슬라이드 2~3장: ① 70% 분해(한계분 vs 사후정보분), ② 사다리 곡선(시점→R² 단조 상승), ③ 결론(누가>무엇>**언제**) + trending_lag 재라벨링
  - **결론 슬라이드**: 4-Analysis 스파인 종결 — "업로드시점 콘텐츠로는 설명 안 됨 / 정확도는 누가>무엇 순, 궁극적으론 언제"
- **완료 기준**: worklog + result + 결론 슬라이드 반영

---

## C. 파일 / 경로

| 항목 | 경로 |
|---|---|
| 입력(base) | `Analysis1_title/LR/out/features_partB_v2.csv` |
| 입력(채널) | `Analysis3_channel/out/step1_channel_features.csv` |
| 입력(사후변수) | `dataset/cleaned_USvideos.csv` |
| 머지 산출 | `Analysis4_final/out/timing_merged.csv` |
| 사다리 결과 | `Analysis4_final/out/timing_ladder_r2.csv` |
| 분해 결과 | `Analysis4_final/out/timing_variance_decomp.csv` |
| 시각화 | `Analysis4_final/out/fig_timing_ladder.png` · `fig_timing_variance.png` |
| 하니스 참조 | `Analysis3_channel/step3_channel_analysis.ipynb`(perf/cv_r2) · `Analysis1_title/LR/r2_ceiling_test.py`(개념) |
| 기록 | `docs/Analysis4_final/Background.md` · `phase1/plan.md`(본 문서) · `phase1/worklog.md` · `phase1/result.md` |

## D. 일정 (D-1)

| Step | 내용 | 소요 | 우선순위 |
|---|---|---|---|
| 0 | 셋업·3원 머지·검증 | 30분 | 필수 |
| 1 | 4단 사다리 하니스 | 1시간 | 필수 |
| 2 | 잔여분산 분해 | 30분 | 필수 |
| 3 | drift & RF 0.245 재라벨링 | 30분 | 권장 |
| 4 | 시각화 | 45분 | 필수 |
| 5 | 문서화·발표 반영 | 45분 | 필수 |

**총 예상**: ~3.5시간 (Drift 포함 ~4시간)

## E. 위험 & 대응

| 위험 | 대응 |
|---|---|
| T0/T1 재현 실패 | **저위험** — A3 `F_content`/`F_chan`·split·category필터 그대로 복제(B표 #2~9). 불일치 시 base 파일·hour_bin Categorical 순서·`dropna(publish_time)` 먼저 점검 |
| 사다리 단조 상승 안 함(T2<T1 등) | finding으로 보고(특정 사후변수 무신호) — trending_lag·참여도는 강신호라 가능성 낮음 |
| T2/T3 R²을 "성과"로 오해 | 모든 보고·그림에 "사후=다른 시점, 시연" 라벨 강제(A표·§5-1) |
| post-upload 머지 결측 | `cleaned_USvideos.csv`는 동일 6249 정본 → `how="left"` 결측 0 기대. 발생 시 video_id 정합성 점검 |
| 잔여분 (1)(2)(3a) 정량 분리 요구 | 범위 외 — 정성 귀속 + 한계 명시(survivorship·sparsity는 데이터 구조라 분해 불가) |
| time-split 여전히 음수 | drift = 시점 의존성 증상으로 흡수, 근본해결 아님 명시(향후 과제) |
| 발표 시간 부족 | Drift 단계 생략 가능. Ladder·Decomposition·Visualization·Documentation이 결론 핵심 |

## F. 성공 기준

- ✅ 4단 사다리 R² **단조 상승** 실측 → "정확도 = 예측 시점의 함수" 수치 입증
- ✅ T0=0.1188, T1=0.2956 재현 → A1~A3와의 연속성·충실성 확인
- ✅ 70% 분해(사후정보 회수분 vs 구조적 잔존분) → A3 evaluation §5-2의 정직한 해부 완성
- ✅ A3 노출 4개 모순(잔여70%·drift·naive-OOF·RF0.245) 정식 해소
- ✅ 4-Analysis 스파인 결론 도출: 누가(채널) > 무엇(콘텐츠) > **언제(시점)**
- ✅ 헤드라인 OOF 0.30 고정 + 사후정보 시연 라벨링 → 정직성 유지

---

**다음 문서**: `docs/Analysis4_final/phase1/worklog.md` — 단계 진행 중 결정·발견·보정 기록
