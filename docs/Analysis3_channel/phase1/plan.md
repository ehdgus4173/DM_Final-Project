# Analysis 3 — Channel Track: 상세 작업계획서 (Execution Plan)

**담당**: Donghyun (Part B 확장 — 콘텐츠 트랙 → 채널 트랙)
**작성**: 2026-06-17, 발표 D-1
**전제 문서**: `docs/Analysis3_channel/Background.md` (노선 진단·근거)
**입력**:
- `dataset/cleaned_USvideos.csv` (n=6249, 26컬럼, 타깃 `log_views`·콘텐츠/시간 피처)
- `dataset/USvideos.csv` (원본 Kaggle, n=40949 — **`channel_title` 복구원**)
- `Analysis1_title/LR/features_partB_v2.csv` (M_content 베이스 피처 정본; `video_id`·`publish_time` 보유)
**작업 위치**: `Analysis3_channel/` (스크립트 + 산출 CSV/PNG)
**하니스 템플릿**: `Analysis1_title/LR/compare_sentiment.py` (M0-vs-M1 평가 구조 재사용)
**환경**: Anaconda `dm`, Python 3.12, VS Code

> **용어**: "Analysis"는 프로젝트 전체 단위(제목=A1, 태그=A2, 채널=A3). "Step"은 채널 트랙 내부 실행 단계(0~6).
> **핵심 방법론 포인트**: out-of-fold(OOF) target encoding — 같은 영상이 자기 타깃을 학습하는 same-video 누수를 K-fold로 차단. 이 트랙의 차별화 지점.

---

## A. 누수(leakage) 스코프 — 비협상

채널 트랙은 **업로드시점에 알 수 있는 정보만** 사용한다.

- **사용**: 콘텐츠 피처(A1 확정) + `C(hour_bin)`/`C(publish_weekday)`/`C(category)` + **채널 피처**(업로드시점)
- **미사용**: `trending_lag`, `days_on_trending`(사후) · `likes`/`dislikes`/`comment_count`(참여도) → **Analysis 4로 이관**
- **channel_title 복구 정당성**: 업로드 시점에 당연히 아는 정보, 정제 때 단순 드롭됐을 뿐 누수 아님. 원본 복구는 정당.
- **채널 평균 누수 차단**: 채널별 평균 `log_views`는 "업로드 전 채널의 과거 실적" 성격 → 시점 정당. 단 same-video leakage(자기 자신이 평균에 포함)는 **OOF로 반드시 차단**. test/CV 수치 중심 보고.

---

## B. 확정된 설계 결정

| # | 항목 | 확정값 | 근거 |
|---|---|---|---|
| 1 | 타깃 | `log_views` | 전 분석 일관 |
| 2 | 베이스라인 M_content | `title_len, caps_ratio, exclaim_cnt, question_cnt, has_number, has_bracket, tag_cnt` + `C(hour_bin)` + `C(publish_weekday)` + `C(category)` | A1 확정 (rand test R²≈0.119) |
| 3 | 채널 피처 ① | 채널별 **OOF 평균 `log_views`** (`chan_mean_oof`) | 채널 규모/팬덤 프록시; OOF로 누수 차단 |
| 4 | 채널 피처 ② | 채널 **등장 빈도** (`chan_freq` = 트렌딩 횟수) | 채널 영향력 프록시 |
| 5 | 단일/희소 채널 fallback | 카테고리 평균 → 글로벌 평균 순 대체 | high-cardinality·single-appearance 처리 |
| 6 | OOF 분할 | **KFold(n=5, shuffle, random_state=42)** | 단일 split 우연성 완화, 재현성 |
| 7 | 평가 | rand split R² + time split R² + **K-fold CV R²(±std)** + ΔR² | A2 Step5 방법론 미러 |
| 8 | 병행 모델 | LR(statsmodels, 계수·해석) + **RF/GBM**(비선형 채널효과 확인) | 채널효과가 비선형일 수 있음 |
| 9 | dedup 패턴 | `sort_values("views").drop_duplicates("video_id", keep="last")` | 전 분석 일관 |

---

## Step 0 — 셋업 & 채널 복구 검증

- **작업**
  - `Analysis3_channel/` 작업 스크립트 셋업, imports: `pandas`, `numpy`, `sklearn.model_selection.KFold`, `statsmodels.formula.api`, `sklearn.metrics`, `matplotlib`
  - 원본 `dataset/USvideos.csv` 로드 → `video_id` 기준 dedup(`drop_duplicates(keep="last")`) → `video_id → channel_title` 매핑 생성
  - 정본 `dataset/cleaned_USvideos.csv` 로드 → `video_id`로 `channel_title` 머지
  - **정합성 점검**: 머지 후 `channel_title` 결측 수, 매칭률(6249 중 몇 개 매칭), 고유 채널 수, 채널당 영상 수 분포(단일 등장 채널 비율)
- **산출**: `Analysis3_channel/step0_channel_merge.csv` (video_id + channel_title + 기존 피처 결합) + 매칭 통계
- **완료 기준**: 매칭률 보고(≈100% 기대), 고유 채널 수·단일 등장 비율 보고 → fallback 규모 파악

## Step 1 — 채널 피처 엔지니어링 (OOF target encoding)

- **작업**
  - `chan_freq`: 채널별 등장 빈도(전체 데이터 count) — 누수 무관(시점 정당한 빈도 프록시)
  - `chan_mean_oof`: KFold(5, shuffle, rs=42)로 각 fold의 **train fold에서만** 채널 평균 `log_views` 계산 → held-out fold에 부여 (자기 타깃 미포함)
  - **fallback**: train fold에 없는(또는 단일 등장) 채널 → 해당 fold의 카테고리 평균 → 없으면 글로벌 평균
  - 비교용 **naive 평균**(`chan_mean_naive`, 전체 데이터로 계산)도 같이 만들어 OOF와 낙관편향 격차 확인(보고용)
- **산출**: `Analysis3_channel/step1_channel_features.csv` (video_id + `chan_mean_oof` + `chan_freq` + `chan_mean_naive`)
- **주의**: `chan_mean_oof`는 모델 평가 split과 무관하게 데이터 전체에 대해 OOF로 한 번 생성(인코딩 누수 차단용). 모델 train/test split은 그 위에서 별도 적용.
- **완료 기준**: OOF vs naive 평균 상관·분포 비교, fallback 발동 건수 보고

## Step 2 — 누수 점검 (방법론 핵심)

- **작업**
  - same-video leakage 차단 검증: `chan_mean_oof`가 자기 영상 타깃을 포함하지 않음을 단일 채널 샘플로 수동 확인
  - naive vs OOF로 각각 M_content+channel 적합 → **naive가 낙관편향(R² 과대)임을 수치로 시연** → OOF 채택 정당화
  - 채널 피처와 `log_views` 상관(예상 높음) + 채널 피처와 콘텐츠 피처 상관(낮음 기대 → 직교 정보)
- **산출**: 누수 점검 요약(naive ΔR² vs OOF ΔR² 격차)
- **완료 기준**: "OOF 없으면 얼마나 부풀려지나"를 수치로 보고

## Step 3 — 모델 비교 (M_content vs M_content+channel)

- **작업** — `compare_sentiment.py` 구조 복제:
  - **M_content** = B-2 베이스 피처 (rand R²≈0.119 재현 확인)
  - **M_content+channel** = M_content + `chan_mean_oof` + `chan_freq`(+ `log1p` 검토)
  - 평가: rand split R²(train/test) + time split R²(drift 확인) + **KFold CV R²(±std)** + ΔR²
  - statsmodels OLS 계수·p-value (채널 피처 유의성·부호)
- **산출**: `Analysis3_channel/step3_model_compare_channel.csv`
  컬럼: `model, rand_R2_train, rand_R2_test, rand_RMSE_test, time_R2_test, cv_R2_mean, cv_R2_std`
- **주의**: time split R²는 채널로 안 풀릴 수 있음(temporal drift는 별개 finding). rand/CV 중심 보고.
- **완료 기준**: ΔR² 보고 + "채널 > 콘텐츠" 수치 입증

## Step 4 — 병행 모델 (RF/GBM) & 비선형 채널효과

- **작업**
  - 동일 피처셋으로 RandomForest/GBM 적합 → test R² + permutation importance
  - 채널 피처가 importance 상위인지 확인(콘텐츠 피처 대비)
  - `tag_cnt`가 채널 피처 추가 후 importance에서 밀려나는지 확인 → "tag_cnt는 채널 프록시였다"(Background §2-3) 가설 검증
- **산출**: `Analysis3_channel/step4_rf_importance.csv` + `Analysis3_channel/step4_fig_importance.png`
- **완료 기준**: 채널 피처 importance 순위 + tag_cnt 변화 보고

## Step 5 — 시각화 & 발표 표

- **작업**
  - ΔR² 막대그래프: M_content → M_content+channel R² 점프 시각화 (4-Analysis 스파인의 ⭐ A3)
  - "분산 절반 잔존" 시각화(설명된 분산 vs 잔여) → A4 피벗 근거
  - permutation importance 그림(채널 vs 콘텐츠)
  - 스타일: A1/A2 팔레트 일관(PRIMARY `#3B4FE4`, ACCENT `#1A7F5A`, GRAY `#C8CDD6`), dpi=200
- **산출**: `Analysis3_channel/step5_fig_channel_r2.png` (+ importance 그림)
- **완료 기준**: R² 점프 + 잔여 분산을 한 그림으로 보여주는 시각화 ≥1

## Step 6 — 문서화 & 발표 반영

- **작업**
  - `docs/Analysis3_channel/phase1/worklog.md`: 진행 중 결정·발견·보정(매칭률, fallback 규모, naive vs OOF 격차, 실제 ΔR², 잔여 분산)
  - `docs/Analysis3_channel/phase1/result.md`: 최종 결과(A2 패턴)
  - 발표 슬라이드 2~3장: ① 왜 채널(콘텐츠 트랙 소진 + tag_cnt 단서), ② R² 점프 그림, ③ "절반 잔존 → A4 피벗"
  - A4 문구는 **실제 숫자 확정 후** 작성(Background §5-1: 0.45든 0.6이든 아크 동일)
- **완료 기준**: worklog + result 작성 + 슬라이드 반영

---

## C. 파일 / 경로

| 항목 | 경로 |
|---|---|
| 입력(정본) | `dataset/cleaned_USvideos.csv` |
| 입력(복구원) | `dataset/USvideos.csv` |
| 입력(베이스 피처) | `Analysis1_title/LR/features_partB_v2.csv` |
| 머지 산출 | `Analysis3_channel/step0_channel_merge.csv` |
| 채널 피처 | `Analysis3_channel/step1_channel_features.csv` |
| 모델 비교 | `Analysis3_channel/step3_model_compare_channel.csv` |
| RF importance | `Analysis3_channel/step4_rf_importance.csv` + `step4_fig_importance.png` |
| 시각화 | `Analysis3_channel/step5_fig_channel_r2.png` |
| 하니스 템플릿 | `Analysis1_title/LR/compare_sentiment.py` |
| 기록 | `docs/Analysis3_channel/Background.md` · `phase1/plan.md`(본 문서) · `phase1/worklog.md` · `phase1/result.md` |

## D. 일정 (D-1)

| Step | 내용 | 소요 | 우선순위 |
|---|---|---|---|
| 0 | 셋업·채널 복구·검증 | 30분 | 필수 |
| 1 | 채널 피처(OOF) 엔지니어링 | 1시간 | 필수 |
| 2 | 누수 점검(naive vs OOF) | 30분 | 필수 |
| 3 | M_content vs +channel 비교 | 1시간 | 필수 |
| 4 | RF/GBM 병행 + importance | 1시간 | 권장 |
| 5 | 시각화·표 | 45분 | 필수 |

**총 예상**: ~4시간(Step 4 포함 ~5.5시간)

## E. 위험 & 대응

| 위험 | 대응 |
|---|---|
| 단일 등장 채널 다수 → 채널 피처 신호 약함 | fallback(카테고리→글로벌 평균) + "R² 기여는 반복 등장 대형 채널 꼬리에서 발생" 명시 (Background §5-4) |
| OOF 누락으로 낙관편향 | naive vs OOF 격차를 Step 2에서 수치 시연, OOF만 헤드라인 |
| R² 점프가 기대(0.4~0.6)보다 작음 | 아크 불변(Background §5-1): "최강 업로드시점 신호도 천장" → A4 피벗 근거로 동일 사용 |
| time split R² 여전히 음수 | temporal drift는 별개 finding으로 유지(채널로 안 풀림 명시) |
| `run_lr.ipynb` 분실로 M_content 재현 불안 | `features_partB_v2.csv`로 재현(산출 CSV 생존), rand R²≈0.119 매칭 확인 |
| Analysis3_channel/.git 0바이트 파일 | 의도치 않은 잔여물 — 작업 전 정리 검토 |

## F. 성공 기준

- ✅ `channel_title` 복구·머지 정합성 확인(매칭률 보고)
- ✅ OOF target encoding으로 same-video 누수 차단 + naive 대비 정직성 시연
- ✅ M_content vs M_content+channel ΔR² 측정 → "채널 > 콘텐츠" 수치 입증
- ✅ 채널을 넣어도 잔여 분산 확인 → **A4 피벗 근거** 확보
- ✅ 기존 A1/A2 폐기 없이 "베이스라인 → 진단 → 채널 돌파 → 천장" 서사로 통합

---

**다음 문서**: `docs/Analysis3_channel/phase1/worklog.md` — Step 진행 중 결정·발견·보정 기록 (Step 0 착수 시 생성)
