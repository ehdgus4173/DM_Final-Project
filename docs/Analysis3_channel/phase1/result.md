# Analysis 3 — Channel Track: 결과 기록 (Result)

> 콘텐츠 트랙(A1 제목·A2 태그)이 공통으로 막힌 **천장을 콘텐츠 너머(채널)로 정면 돌파**하는 단계.
> "조회수는 *무엇을* 올렸나(콘텐츠)가 아니라 *누가* 올렸나(채널)가 지배한다"를 누수 없는 업로드시점 피처로 수치 입증.
> (배경·근거는 `Background.md`, 실행계획은 `plan.md`, 진행기록은 `worklog.md`)

---

## 0. 한 줄 요약

> **채널이 콘텐츠를 압도한다 — 그러나 천장은 남는다.** 업로드시점 채널 피처(OOF 채널평균 + 등장빈도)를 넣자 설명력이 **CV R² 0.107 → 0.323 (ΔR²=+0.216), rand test 0.119 → 0.296으로 ~2.7배 점프**. RF importance에서도 채널이 1·2위를 독식하고, A1에서 최강이던 `tag_cnt`는 1위→3위로 밀려 **"콘텐츠 신호처럼 보이던 게 실은 채널 프록시였다"**가 직접 확인됨. 그럼에도 **분산의 70%가 여전히 미설명** — 최강 업로드시점 신호조차 절반 이상을 못 푼다는 사실이 **A4(예측 시점 재프레이밍)의 근거**가 된다.

---

## 1. 목적 & 설계

- **가설 H_A3**: 조회수는 콘텐츠가 아니라 채널이 지배한다 → 채널 정체성을 업로드시점 피처로 넣으면 R²가 점프한다("배경 > 콘텐츠").
- **입력**: `dataset/cleaned_USvideos.csv`(타깃·콘텐츠/시간) + `dataset/USvideos.csv`(channel_title 복구) + `Analysis1_title/LR/features_partB_v2.csv`(M_content 베이스).
- **누수 차단**(비협상): 업로드시점 변수만. `trending_lag`/`days_on_trending`/참여지표 전면 배제 → A4 이관.
- **채널 피처**:
  - `chan_mean_oof` = KFold(5) **out-of-fold** 채널 평균 `log_views` (same-video 누수 차단)
  - `chan_freq_log` = `log1p`(채널 등장빈도) — 영향력 프록시
  - fallback: 희소/단일 채널 → 카테고리 평균 → 글로벌
- **모델**: M_content vs M_content+channel. statsmodels OLS(계수·p) + sklearn KFold CV(±std) + rand/time split + RF/permutation 병행.

---

## 2. 결과

### 2-1. 채널 복구 정합성 (`step0_channel_merge.csv`)

- 매칭률 **100%**(6249/6249), 고유 채널 **2,100**, 단일 등장 **61.6%**(high-cardinality).

### 2-2. 모델 비교 (`step3_model_compare_channel.csv`)

| model | rand R² (train/test) | RMSE test | time R² | CV R² (mean±std) |
|---|---|---|---|---|
| **M_content** | 0.119 / **0.1188** | 1.752 | −1.271 | 0.1067 ± 0.0168 |
| **M_content+chan (OOF)** | 0.342 / **0.2956** | 1.566 | −0.568 | **0.3229 ± 0.0207** |
| M_content+naive *(누수)* | 0.754 / 0.7366 | 0.958 | 0.362 | 0.7481 ± 0.0218 |
| **Δ (OOF − content)** | — / **+0.1768** | −0.186 | +0.703 | **+0.2162** |

- M_content rand test R²=**0.1188** → A1 베이스라인과 정확히 일치(재구성 충실성 확인).
- **ΔR²(CV)=+0.216, std≈0.02** → 0이 ±std 밖 = 견고한 개선(태그·감성의 ΔR²≈0과 정반대).
- **naive vs OOF 격차 +0.44** = same-video 누수 낙관편향. OOF만 헤드라인(정직성).

### 2-3. 채널 계수 (OLS, `compare_channel.py`)

| feature | coef | p | 해석 |
|---|---|---|---|
| `chan_mean_oof` | **+0.700** | <.001 | 채널 과거 평균이 1 오르면 log_views +0.70 (지배 신호) |
| `chan_freq_log` | **+0.244** | <.001 | 자주 트렌딩되는 채널일수록 ↑ (영향력) |
| `tag_cnt` | +0.0104 | <.001 | **채널 통제 후 0.0218→0.0104로 반감** |

### 2-4. RF / permutation importance (`step4_*`)

- RF test R²: 0.112 → **0.304** (ΔR²=+0.192, LR과 일관 → 채널효과는 주로 선형).
- importance: **`chan_mean_oof` 1위(0.525, 압도) > `chan_freq_log` 2위 > `tag_cnt` 3위**(총 33피처).
- **`tag_cnt` 순위: M_content 1위 → +channel 3위** → A1 RF의 최강 업로드시점 변수가 채널 뒤로 밀림.

---

## 3. 해석

1. **"배경 > 콘텐츠"가 수치로 입증됨.** 제목·태그·감성이 못 더한 예측력(ΔR²≈0)을, 채널 단 2개 피처가 **ΔR²=+0.216(CV)**로 더함. 조회수의 일차 결정자는 콘텐츠 표면이 아니라 "누가 올렸나"다.

2. **`tag_cnt`의 정체 규명 (관통 단서 해소).** A1 RF에서 "업로드시점 최강"이던 `tag_cnt`는 채널을 통제하자 계수 반감·importance 3위로 강등 → **콘텐츠 신호가 아니라 채널 운영수준/제작 정성의 약한 프록시**였음(Background §2-3 가설 확정). A1·A2를 관통하던 의문에 답을 줌.

3. **그러나 분산의 70%가 미설명 — 천장 확인.** 최강 업로드시점 신호(채널)를 넣어도 rand test R²는 0.30에 그침. 나머지는 구독자 수·추천 알고리즘·외부 홍보처럼 **업로드시점 데이터에 없거나 사후에만 관측 가능**한 요인. → 더 정확히 예측하려면 "무슨 콘텐츠 피처냐"가 아니라 **"언제 예측하느냐"를 바꿔야 한다**(A4).

4. **정직성 — 0.30은 실패가 아니다.** 업로드시점 정보만으로 R²≈0.30은 사실 준수한 수준. 피벗 논리는 "0.30이 나쁘다"가 아니라 **"최강 신호도 절반 이상을 못 푼다 → 한계는 콘텐츠가 아니라 예측 시점"**(Background §5-1, 숫자와 무관하게 아크 동일).

---

## 4. 한계

- **high-cardinality + 단일 등장 61.6%**: R² 기여는 반복 등장 대형 채널의 고조회수 꼬리에서 주로 발생. 단일 등장 채널은 카테고리 평균 fallback(23%)으로 회귀 → 약신호.
- **구독자 수 부재**: 데이터셋에 없어 채널 과거 평균을 프록시로 사용. 직접 추정/외부 API는 미수행(범위 명시).
- **temporal drift(time-split R²=−1.27→−0.57)는 채널로 안 풀림** — 별개 finding 유지.
- **naive 버전 낙관편향(+0.44)**: OOF 필수성의 반증이자 방법론 포인트. 헤드라인은 OOF만.

---

## 5. 발표 반영 (권고)

- **A3 = 4-Analysis 스파인의 ⭐ 클라이맥스 + A4 발판.** 슬라이드 2~3장:
  1. **왜 채널인가**: 콘텐츠 트랙 소진(ΔR²≈0) + `tag_cnt` 단서 → "누가 올렸나를 안 봤다".
  2. **R² 점프 그림**(`step5_fig_channel_r2.png` 좌) + importance(`step4_fig_importance.png`): 채널 1·2위, tag_cnt 강등.
  3. **절반의 천장**(`step5_fig_channel_r2.png` 우) → "최강 신호도 70% 못 푼다 → A4: 예측 시점".
- **OOF/누수 차단을 방법론 셀링포인트로**: "naive면 0.74로 부풀지만 정직하게 OOF 0.30 보고" → 단순 회귀 프로젝트와 차별화.
- 톤: "채널이 콘텐츠를 이긴다"는 강하게, "그래도 천장"은 A4로 자연스럽게 연결.

---

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `Analysis3_channel/build_channel_features.py` | Step 0+1: 복구·머지 + OOF 인코딩 |
| `Analysis3_channel/compare_channel.py` | Step 2+3: 누수 점검 + 모델 비교 |
| `Analysis3_channel/rf_channel.py` | Step 4: RF + permutation importance |
| `Analysis3_channel/viz_channel.py` | Step 5: 발표 시각화 |
| `step0_channel_merge.csv` | channel_title 머지본 (6249×27) |
| `step1_channel_features.csv` | 채널 피처(OOF/naive/freq) |
| `step3_model_compare_channel.csv` | 모델별 R²/RMSE/CV |
| `step4_rf_importance.csv` + `step4_fig_importance.png` | RF permutation importance |
| `step5_fig_channel_r2.png` | R² 점프 + 잔여분산 (발표용) |

---

## 7. 다음 (A4 핸드오프)

- **A4(예측 시점 재프레이밍)**: 잔여 70%·temporal drift·naive vs OOF 모순·RF 헤드라인 R²=0.245(trending_lag 포함) 모순 → "정확도는 콘텐츠가 아니라 예측 시점의 함수"로 정식 해소.
- **A4 문구는 본 결과 확정 후 작성 완료** 가능: 실제 OOF R²=0.30, 잔여 0.70 기준으로 "최강 업로드시점 신호의 천장" 서사 확정.
- **관통 finding H4(카테고리 이질성)** 유지: 채널 트랙은 pooled로 진행했으나 카테고리 더미 통제 하에 채널효과 입증.
