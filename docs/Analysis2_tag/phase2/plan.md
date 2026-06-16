# Phase 2 — LR 재설계 계획 & 과제 검토 방법론

> 분석(Phase 1) → 모델로 루프를 닫는 단계. Step 5 = 태그 효과를 LR에 통합한 재설계.
> 과거 과제(`Assignment01_ITM` = Ames Housing LR, `Assignment02_ITM` = Adult Income 분류·모델비교) 검토에서
> 이번 케이스에 적용할 방법론을 추출해 반영. **다음 세션 context 핸드오프 문서.**

---

## 0. 작업 공간

- 노트북: `Track2/LR/` (Phase 2 신규 작업). 파일명은 `step5_*` prefix.
- 입력:
  - `Track1/LR/features_partB_v2.csv` — 기존 LR feature + `publish_time` 보유
  - `Track2/phase1/step1_tags_parsed.csv` — 태그 이진 생성용 (video_id로 머지)
- 하니스 템플릿: `Track1/LR/compare_sentiment.py` (M0-vs-M1 평가 재사용)

## 1. 확정 설계 (둘 다 권장안 채택)

- **모델 비교: pooled M0 vs M_tags** (카테고리별 분리 X — 직접 비교·소표본 안정성)
- **태그 셋: FDR 생존 7개** (보수적)
  - `is_disney, is_movie, is_trailer, is_pop, is_country, is_music, is_alternative`
- 베이스라인 확장 방식(기존 LR을 갈아엎지 않음):
  - **M0** = obj feats + `C(hour_bin)` + `C(publish_weekday)` + `C(category)` (rand test R²≈0.1188)
  - **M_tags** = M0 + 위 태그 7개 이진
  - **M_lasso**(선택) = 후보 태그 전부 투입 → CV로 자동 선택 (아래 §2-4)

## 2. 과제 검토에서 도입할 방법론

### 바로 적용 (3)

**(1) K-fold 교차검증** — *출처: 과제2 StratifiedKFold*
- 회귀용 `KFold`로 변환. 단일 rand-split 대신 **CV 평균±표준편차 R²**로 M0/M_tags 비교.
- 효과: 단일 분할의 우연성 + selection bias 우려를 정면 완화.
- 보고 톤: "ΔR² = +0.00x ± 0.0xx (분산 안에 0 포함 여부)".

**(2) 왜도 체크 → 치우친 예측변수 로그변환** — *출처: 과제1 §3.2.2~3.2.3*
- `skew > 0.75(abs)`이면 `np.log1p`. 후보: `exclaim_cnt`, `question_cnt`, `tag_cnt` (count라 우편향 가능).
- **트레이드오프**: 계수 해석이 "1 증가"→"1% 증가"로 바뀜 → H4 해석 서사와 충돌 가능. 적용 전후 R²·해석성 비교 후 채택.

**(3) permutation_importance** — *출처: 과제2*
- **test-set**에서 모델-비의존 out-of-sample 중요도. statsmodels 계수(in-sample)를 보완.
- 용도: "태그 이진이 표본 밖에서 실제 기여하나"를 정직하게 검증/시각화.

### 조건부 (검토 추천)

**(4) Lasso 자동 선택** — *출처: 과제2 GridSearchCV(α 튜닝) 발상*
- 후보 태그를 **전부 투입** → Lasso가 CV로 선택. **사전 선택(double-dipping) 편향 제거.**
- OLS p-value 해석은 잃지만 "데이터가 스스로 고른 태그"라는 더 강한 서사 획득.
- → M0 / M_tags(사후선택) / M_lasso(자동선택) 3종 비교 가능.

**(5) VIF 재확인** — *출처: 과제1*
- 태그 이진 + `C(category)` 조합 다중공선성 재점검 (disney가 Film 전용 → 카테고리 더미와 중첩 우려).

### 아키텍처
- **statsmodels(계수·p-value) + sklearn KFold(CV R²) 병행.** 과제1·2 기법을 동시에 사용.
- 재현성: `RANDOM_STATE=42` 고정, M0/M_tags 동일 분할.
- (선택) 무태그 244개에 `has_no_tag` 플래그 — 과제2 "Unknown 범주" 발상의 약한 차용.

## 3. 적용하지 않음 (이유 명시)

| 과제 기법 | 배제 이유 |
|-----------|-----------|
| Pipeline + ColumnTransformer로 **sklearn LogisticRegression 전환** | 우리 LR은 **p-value·계수 추론(H4)이 핵심** → statsmodels 유지. Pipeline은 Part C(RF)에 적합. |
| 분류 지표(accuracy/precision/recall/f1/roc_auc), class imbalance | 회귀라 무관. 우린 R²·RMSE·MAE. |
| GridSearchCV 하이퍼파라미터 튜닝 | OLS엔 튜닝할 파라미터 없음. (정규화로 가면 Lasso α만 — §2-4) |

## 4. 정직성 플래그 (반드시 보고에 반영)

- **기대 R² 상승 작음**: 감성(VADER) 추가가 ΔR²≈−0.002였음. 메타데이터 천장 낮음. 태그도 헤드라인 뒤집을 정도 아닐 것.
- **selection bias / double-dipping**: 태그를 전체 데이터 유의성으로 골라 같은 데이터에 재투입 → 낙관 편향. test/CV 수치 중심 보고 + "태그 사후 선택됨" 명시. (Lasso로 근본 완화 가능)
- **temporal drift(time-split R²=−1.28)는 태그로 안 풀림** — 별개 finding으로 유지.
- 가치 = "예측력 대박"이 아니라 **"tag_cnt 너머 내용이 신호를 더하나 + 한계"** 입증. 프로젝트 '천장 낮음' 서사와 정합.

## 5. 산출 예정

- `Track2/LR/step5_tag_analysis.ipynb` — M0/M_tags(/M_lasso) CV 비교 + 계수표 + permutation importance.
- 결과 → `docs/2/phase2/result.md` (Phase 1과 동일 패턴).
