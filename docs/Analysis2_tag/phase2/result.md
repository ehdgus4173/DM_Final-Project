# Phase 2 — Step 5: 태그 효과 LR 통합 결과 기록

> 분석(Phase 1, 단변량) → 모델(다변량 LR)로 **루프를 닫는** 단계.
> Step 5 = FDR 생존 7개 태그를 기존 LR 베이스라인에 통합해, "태그 **내용**이 `tag_cnt`(개수) 너머 신호를 더하는가"를 다변량으로 검증.
> (계획·과제 방법론 검토는 `plan.md`, Phase 1 결과는 `../phase1/result.md`)

---

## 0. 한 줄 요약

> **예측 payoff는 사실상 0** (ΔR² ≈ noise). 단, 다변량 통제가 **해석을 날카롭게** 함 — Film의 흥행 태그(disney/movie/trailer)는 카테고리 효과의 교란으로 소멸하고, Music 태그(pop↑, music/country/alternative↓)만 통제 후에도 살아남음. 결론은 "대박 발견"이 아니라 **modest confirmatory null**: 프로젝트 thesis("단일 메타데이터로는 부족")를 닫고 방법론 가치를 남기는 마지막 조각.

---

## 1. 목적 & 설계

- **입력**: `Track1/LR/features_partB_v2.csv`(기존 LR feature) + `../phase1/step1_tags_parsed.csv`(태그, `video_id` 머지)
- **노트북**: `Track2/phase2/step5_tag_analysis.ipynb` (11셀, `dm` 커널 실행)
- **누수 차단**: 업로드시점 변수만. `trending_lag`/`days_on_trending`/참여지표 전면 배제.
- **모델 (pooled, 카테고리 분리 X)**
  - **M0** = obj feats(`title_len, caps_ratio, exclaim_cnt, question_cnt, has_number, has_bracket, tag_cnt`) + `C(hour_bin)` + `C(publish_weekday)` + `C(category)` — `word_count`은 `title_len`과 다중공선성으로 제외(기존 결정 유지). **`tag_cnt` 포함**이 핵심: M_tags가 "개수 너머 내용"을 검증하는 구조.
  - **M_tags** = M0 + FDR-7 이진(`is_disney, is_movie, is_trailer, is_pop, is_country, is_music, is_alternative`)
- **평가** (plan §2 필수 3종)
  - statsmodels OLS → 태그 계수·p (다변량 통제 후 부호/유의성)
  - sklearn 5-fold CV R² (±std) + ΔR² — 단일 분할 우연성 완화
  - rand-split test R² (compare_sentiment M0-vs-M1 미러링)
  - test-set permutation importance — 표본 밖 기여
- **미적용 (deferred, plan §1·§2-4)**: Lasso 자동선택, 왜도 로그변환, `has_no_tag` 플래그. 왜도는 체크·출력만 하고 해석성 위해 모델은 선형 유지.

---

## 2. 결과

### 2-1. 모델 비교 (`step5_model_compare_tags.csv`)

| model | n_feat | in-sample R² | CV R² (mean±std) | rand test R² |
|---|---|---|---|---|
| M0 | 31 | 0.1213 | 0.1067 ± 0.0168 | **0.1188** |
| M_tags | 38 | 0.1280 | 0.1109 ± 0.0194 | 0.1210 |
| Δ (M_tags−M0) | +7 | +0.0067 | **+0.0042** | +0.0023 |

- M0 rand test R² = **0.1188** → 문서상 베이스라인과 정확히 일치 (M0 재구성 충실성 확인).
- **ΔR²(CV) = +0.0042, std ≈ 0.018** → 0이 ±std 안에 있음 = **예측 개선은 noise 수준**. 감성(VADER) ΔR²≈−0.002와 같은 "메타데이터 천장 낮음" 결론. (감성은 음수, 태그는 미세 양수지만 둘 다 무의미.)

### 2-2. 태그 계수 — 다변량 통제 후 (`step5_tag_coefs.csv`)

카테고리·시간·객관지표 통제 시, 단변량(Step 3) 효과가 **둘로 갈림**:

| 태그 (출처 카테고리) | 단변량 diff (Step 3) | 다변량 coef (p) | exp(배수) | 판정 |
|---|---|---|---|---|
| Film: **disney** | +1.46 (`***`) | +0.312 (p=.148) | 1.37 | **소멸** |
| Film: **movie** | +0.99 (`*`) | +0.006 (p=.970) | 1.01 | **소멸** |
| Film: **trailer** | +0.93 (`***`) | +0.125 (p=.345) | 1.13 | **소멸** |
| Music: **pop** | +0.59 (`*`) | +0.343 (p=.013, `*`) | 1.41 | 생존 (+) |
| Music: **alternative** | −1.16 (`*`) | −0.847 (p=.0008, `***`) | 0.43 | 생존 (−) |
| Music: **country** | −0.78 (`*`) | −0.802 (p=.0019, `**`) | 0.45 | 생존 (−) |
| Music: **music** | −0.90 (`*`) | −0.488 (p<.001, `***`) | 0.61 | 생존 (−) |

### 2-3. Permutation importance (test set, `step5_fig_perm_importance.png`)

`is_music`(≈0.008) > `is_country` > `is_pop` 순으로 양의 기여, `is_alternative` 미세 양수. `is_movie`·`is_trailer` ≈ 0, `is_disney` 미세 음수. → 계수표와 동일: 표본 밖에서도 Music 태그만 기여, Film 태그는 기여 없음.

---

## 3. 해석

1. **예측 null이 1차 결론.** 태그 내용은 `tag_cnt` 너머 예측 신호를 더하지 못함(ΔR²≈0). 프로젝트의 "조회수 결정요인의 대부분은 업로드시점 메타데이터 밖에 있다"는 thesis를 재확인.

2. **다변량 통제가 해석을 날카롭게 함 (Step 3이 못 한 부분).**
   - **Film 태그 3개 전부 유의성 소멸**: 단변량 "disney ≈4.3배" 효과는 태그 효과가 아니라 **Film 카테고리 효과의 교란**이었음을, 카테고리 더미를 통제한 모델로 직접 입증. plan의 "disney = IP/브랜드 교란, 상관≠인과" 경고를 정식 검증.
   - **Music 태그 4개 전부 부호·유의성 유지**: 카테고리 너머의 within-pool 신호. 흔한/포괄적 태그(music/country/alternative) ↓, pop ↑ = 선택효과 해석이 다변량에서도 견고.

3. **단, "발견"으로 과대평가 금지 (정직성).**
   - disney 소멸은 **상당 부분 구조적**: disney가 Film 전용 태그라 `C(category)=Film` 더미와 공선 → 계수 흡수가 데이터 구조상 예견된 결과. 놀라운 인과 발견은 아님.
   - Music 태그 생존은 **사후선택(double-dipping) 편향**이 껴 있음: 같은 데이터로 FDR 선별한 태그를 같은 데이터에 재투입 → 낙관 편향. 편향 없는 버전은 deferred한 **Lasso 전체 후보 투입**.

4. **종합 판정: modest confirmatory null.** "빈손"이 아니라 well-posed 질문에 대한 **음성 결과 + 루프 종료**. DM 과목 관점의 가치는 산출 수치가 아니라 **방법론** — 누수 차단, 다변량 통제, CV+permutation 병행, 골라낸 결과가 아닌 **정직한 null 보고**.

---

## 4. 한계

- **double-dipping**: 태그가 전체 데이터 유의성으로 사후선택됨 → Music 생존의 독립성 약함. (근본 완화 = Lasso)
- **ΔR² noise 수준**: 예측 개선 없음. exp(배수)는 단변량 해석용이며 통제 후 Film은 비유의.
- **disney 교란은 구조적**: 카테고리 더미와의 공선에 기인.
- **temporal drift(time-split R²=−1.28)는 태그로 안 풀림** — 별개 finding으로 유지.
- 단변량→다변량 통제는 했으나, **편향 없는 자동 선택(Lasso)은 미실행**.

---

## 5. 발표 반영 (권고)

- **발견으로 헤드라인 걸지 않기.** 베이스라인 → 한계 → 피벗(태그) → **닫힌 루프**의 마지막 조각으로 압축.
- **슬라이드 1장**: "태그 내용도 검증 → 천장 동일(ΔR²≈0), 단 카테고리 교란 1건(disney) 확인." 그래프 1개(permutation importance, 영어 라벨) + 계수표 핵심 행.
- 톤: "통계적 유의 ≠ 실무적 큰 효과", "상관≠인과" 일관 유지.

---

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `step5_tag_analysis.ipynb` | M0/M_tags 비교 + OLS 계수 + CV + permutation (11셀) |
| `step5_model_compare_tags.csv` | 모델별 in-sample/CV/rand test R² + Δ |
| `step5_tag_coefs.csv` | M_tags 태그 7개 계수·std_err·p·sig·exp |
| `step5_fig_perm_importance.png` | test-set permutation importance (영어 라벨, dpi=200) |

---

## 7. 다음 (deferred / Phase 4)

- (선택, 시간 허용 시) **Lasso 전체 후보 투입** → double-dipping 제거한 "데이터가 스스로 고른 태그" 버전. M0 / M_tags(사후선택) / M_lasso(자동선택) 3종 비교 가능.
- Part A(EDA) · Part C(RF)와 **Phase 4 머지**.
- **누수 경계 팀 통일**: RF 헤드라인 R²=0.245가 `trending_lag`(사후) 포함값 → 누수 서사와 모순 가능. 태그 트랙은 업로드시점 원칙 준수로 진행됨.
