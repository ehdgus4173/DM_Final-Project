# Part B — Phase 1 작업일지

**담당**: Donghyun (Part B — Feature Engineering & Linear Regression)
**가설**: H4 — 제목 스타일(객관 지표)이 조회수에 영향을 주며, 그 효과는 카테고리마다 다르다
**성격**: 팀 4단계 합류 전 **혼자 도는 셀프 피드백 루프** (1차 구현 → 진단 → 보정 → 마무리)
**입력**: `EDA/cleaned_USvideos.csv` (A 산출, n=6249) / `EDA/USvideos.csv` (원본, publish_time 출처)

---

## 0. 전체 흐름 한눈에

```
run_lr.ipynb (1차 구현)
   ↓  TEST R²=−1.28 발견
diag_test_r2.ipynb (우선순위 1: 음수 원인 진단)
   ↓  원인 = temporal drift
refit_vif.ipynb (우선순위 2: VIF 처리 + 재적합)
   ↓  공선성 해소, 결론 불변
resid_diag.ipynb (우선순위 3: 잔차 마무리)
   ↓  위반 경미, 추가 변환 불필요
[셀프 루프 종료 → 4단계 팀 합류 준비 완료]
```

---

## 1. run_lr.ipynb — 1차 구현

**무엇을**: 작업계획서의 1~3단계를 노트북으로 구현.
- 1단계 Feature Engineering: 객관적 제목 파생변수 8종(title_len, word_count, caps_ratio, exclaim_cnt, question_cnt, has_number, has_bracket, tag_cnt) + publish_hour 4구간 원-핫
- 2단계 Baseline OLS: 전체 데이터 statsmodels OLS + 통계검증 4종(계수·p / VIF / 잔차·QQ / R²·RMSE)
- 3단계 카테고리별 분리 회귀: Top3 각각 OLS → 계수 비교표

**왜**: 누수 없는 업로드시점 변수만으로 "자극성" 베이스라인을 세우고 H4를 검증하기 위해. 통계 해석이 목적이라 RF가 아닌 OLS 선택.

**보정 결정 2가지** (계획서에 미명시였던 부분):
- 시간 split 키: cleaned에 날짜 없음 → 원본에서 `publish_time` 병합해 split 기준으로 사용
- 중복 영상: video_id 기준 dedup (실제로는 중복 0건이었음)

**결과**:
- train R²=0.113 (낮은 설명력 = 의도된 베이스라인)
- 유의 변수(+): caps_ratio, has_bracket, exclaim_cnt, tag_cnt, word_count / 유의(−): title_len / 무의미: question_cnt, has_number
- **VIF 경고**: title_len 7.77 ↔ word_count 7.76 (공선성)
- **TEST R² = −1.28** ⚠️ (시간 split) — 평균 예측보다 나쁨
- Top3 = Entertainment / Music / Howto & Style (계획서 가정 Gaming/Film&Animation과 불일치)
- H4: has_number·exclaim_cnt 부호 반전으로 지지 증거 확보
- 산출: `features_partB.csv`, `coef_comparison_top3.csv`, `fig_lr_residuals.png`

---

## 2. diag_test_r2.ipynb — 우선순위 1: TEST R² 음수 진단

**왜**: 음수 R²는 "모델 결함"일 수도, "데이터 특성"일 수도 있어 원인 규명이 최우선.

**무엇을**: 4갈래 진단 — A 분포 변화 / B 미학습 카테고리 / C 극단 잔차 / α 랜덤 split 비교.

**결과 — 원인 = temporal drift(타깃 분포 이동):**

| 진단 | 결과 | 판정 |
|---|---|---|
| A 분포 변화 | train 평균 12.73 → test 14.30 (+1.56), 표준편차 1.80→1.22 | **주범** |
| B 미학습 카테고리 | only_test=∅, only_train=∅ | 무관 |
| C 극단 잔차 | 상위 1% 제거 시 R² −1.28→−1.35(악화), 잔차 상위 전부 양수 | 무관 (체계적 under-prediction) |
| α 랜덤 vs 시간 | 랜덤 R²=**+0.116**(정상) vs 시간 R²=**−1.28** | **결정적 증거** |

→ **버그가 아닌 발견(finding).** 뒤 시기(2018-03~06) 트렌딩 영상이 체계적으로 고조회수 → 모델이 일괄 과소예측.

**조치 확정**: (a) 성능은 랜덤 split(+0.12) 병기로 정상 베이스라인 확보, (b) 시간 split 음수는 4단계(ET 변환·드리프트) 스토리 근거로 보고. split 보정은 불필요(B=∅).

---

## 3. refit_vif.ipynb — 우선순위 2: VIF 처리 + 재적합

**왜**: title_len↔word_count 공선성이 계수 해석을 왜곡할 수 있어 제거 후 결론이 유지되는지 확인.

**무엇을**: word_count 제거(title_len 유지) → VIF 재확인 / 계수 제거 전후 비교 / 성능 시간·랜덤 병기 / H4 재적합.

**결과**:
- **VIF 해소**: title_len 7.77→**1.09**, 전 변수 1.0~1.1대
- **계수 안정**: 주요 변수 방향·유의성 유지(caps_ratio 0.43·has_bracket 0.39·exclaim_cnt 0.17·tag_cnt 0.02), title_len −0.014→−0.007(여전히 유의) → **제거가 결론을 바꾸지 않음**
- **성능**: 시간 TEST −1.28 / 랜덤 TEST +0.115, R² 손실 없음(0.113→0.112)
- **H4 더 선명**: has_number(Music만 −0.47*), exclaim_cnt(Music만 −0.32), caps_ratio(Music만 유의X)
- 산출: `coef_comparison_top3_v2.csv`
- **확정 feature set**: title_len, caps_ratio, exclaim_cnt, question_cnt, has_number, has_bracket, tag_cnt

---

## 4. resid_diag.ipynb — 우선순위 3: 잔차 진단 마무리

**왜**: OLS 추론 타당성(등분산·정규성) 점검. 1차 잔차 그림은 word_count 포함 모델 기준이라 확정 모델로 재작성 필요.

**무엇을**: 확정 모델 잔차/QQ 재작성 + 정량 검정(Breusch-Pagan, Jarque-Bera, 왜도·첨도, 표준화잔차 꼬리).

**결과**:
- 등분산성 BP p≈8e-25 기각 / 정규성 JB p≈2e-34 기각 — **둘 다 n=4999 대표본 특성**(작은 위반도 기각)
- 실질 크기는 경미: 왜도 −0.39, 첨도 3.35(정규 0/3에 근접), |z|>2 5.44%(기대 4.6%)
- **추가 변환 불필요** — log1p로 충분. 엄밀성 원하면 robust SE(HC3) 옵션
- 산출: `fig_lr_residuals_v2.png`

---

## 5. 종합 결론 (Phase 1 성과)

1. **누수 없는 객관 제목 베이스라인 확보** — R²는 낮지만(랜덤 0.12) 의도된 출발점
2. **TEST 음수는 temporal drift로 규명** — 모델 결함 아님, 4단계 스토리의 다리
3. **공선성 제거 후에도 결론 견고** — 해석 신뢰 가능
4. **잔차 가정 위반 경미** — 모델 형태(OLS+log1p) 적정
5. **H4 지지** — 자극성 효과가 카테고리에 의존(특히 Music에서 부호 반전)

---

## 6. Phase 2(ET) 데이터 반영 — 전체 재실행 (2026-06-17)

**계기**: Part A가 `EDA_Phase2/cleaned_USvideos.csv` 인계. 변경점 — (a) `publish_hour`/`publish_weekday`가 **UTC→ET(DST 자동처리)**, (b) 제목 파생변수 8종·`hour_bin`이 정제본에 **이미 포함**, (c) 데이터가 `EDA/EDA_Phase1·EDA_Phase2/`로 재배치(구 `EDA/cleaned_USvideos.csv` 삭제).

**조치**:
- `run_lr.ipynb` 입력 경로 교체 → `EDA_Phase2/cleaned_USvideos.csv`(데이터) + `EDA_Phase1/USvideos.csv`(publish_time, 시간 split 정렬키로만 사용). 시간 변수는 ET 기준으로 자동 반영.
- 노트북 4개 전부 재실행(`features_partB.csv` ET 기준 재생성 → diag/refit/resid 갱신).

**데이터 영향**: UTC→ET로 `hour_bin` 라벨의 **76.7%**, `publish_weekday`의 **13.5%**가 이동.

**결과 — 결론 대부분 견고, 시간 변수에서 유의미한 변화 1건**:

| 항목 | UTC (Phase1) | ET (Phase2) | 판정 |
|---|---|---|---|
| 제목변수 계수(확정 모델) | caps 0.43·bracket 0.39·exclaim 0.16·tag 0.022·title_len −0.007 | caps 0.42·bracket 0.39·exclaim 0.16·tag 0.022·title_len −0.007 | 사실상 불변 |
| VIF | title_len↔word_count 7.77 → 제거 후 1.09 | 동일 | 불변 |
| 성능 | 랜덤 +0.115 / 시간 −1.28 | 랜덤 **+0.119** / 시간 **−1.277** | 불변 |
| 잔차(BP/JB/왜도/첨도) | 8e-25 / 2e-34 / −0.39 / 3.35 | 1.4e-25 / 1.6e-33 / −0.39 / 3.34 | 불변(경미) |
| **요일 변수** | 전 요일 무의미 | **Sat +0.239\*(p=.025), Sun +0.266\*(p=.009)** | **ET에서 주말 유의해짐** |
| hour_bin | 전 구간 무의미 | 전 구간 무의미 | 불변 |
| H4(카테고리별) | has_number Music만 −, exclaim Music만 − | has_number Music만 −0.476\*, exclaim Music만 −0.336, caps 전 그룹 유의 | **H4 더 선명** |

→ **핵심 발견**: ET 변환이 LR에서도 **요일(주말) 효과를 유의하게** 만듦 — Part A 핸드오프의 "ET 변환 후 weekday 유의성 강화(p 3.1e-5→8.3e-8)"가 LR 계수에서도 재현됨. drift(시간 split 음수)는 ET로도 유지되어 시간대 편향만으로 설명되지 않음(수집기간 편중 요인 잔존).

**산출 갱신**: `features_partB.csv`(ET), `coef_comparison_top3_v2.csv`(ET), `fig_lr_residuals.png`/`fig_lr_residuals_v2.png` 재생성.

---

## 7. 다음 작업 (4단계 — 팀 합류)

**가져갈 인수인계 포인트**:
- 확정 feature set (word_count 제외)
- 성능 보고 규칙: 랜덤 split = 베이스라인, 시간 split 음수 = drift 발견
- 발표 스토리 골격: 베이스라인 → 한계(drift·낮은 설명력) → 개선

**개선 항목**:
1. **ET 시간 변환** — 데이터셋이 UTC라 미국 시청자 체감 시간과 어긋남 → ET 변환 후 재분석. drift 일부가 시간대 편향에서 비롯됐는지 점검과 연결.
2. **감성 파생변수(VADER 등)** — 객관 지표만으론 설명력 약함 → 감성 신호로 보완.
3. (옵션) 시간 split 보고 시 robust SE 병기 검토.

**열린 질문**:
- drift의 정체(수집기간 큰 영상 편중 vs 시간대 편향)를 ET 변환 후 분해 가능한지
- 감성변수 추가 시 R² 개선 폭이 발표에서 "개선" 서사로 충분한지

---

## 산출물 목록 (LR/)
- `run_lr.ipynb` — 1차 구현
- `diag_test_r2.ipynb` — TEST 음수 진단
- `refit_vif.ipynb` — VIF 처리 + 재적합
- `resid_diag.ipynb` — 잔차 마무리
- `features_partB.csv` / `coef_comparison_top3.csv` / `coef_comparison_top3_v2.csv`
- `fig_lr_residuals.png` / `fig_lr_residuals_v2.png`

## 문서 (docs/)
- `PartB_작업계획서.md` — 원 설계
- `PartB_1차검토_계획서.md` — 결과 검토 + 우선순위별 진행기록
- `phase1_작업일지.md` — 본 문서
