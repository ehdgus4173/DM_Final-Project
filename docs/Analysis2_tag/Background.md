# Phase 2 — Tag-Based Category Analysis: Background

**담당**: Donghyun (Part B 확장)
**작성**: 2026-06-17, 발표 D-1
**입력**: `Track1/EDA/EDA_Phase2/cleaned_USvideos.csv` (Part A 인계, n=6249, tags 컬럼 포함)
**산출 위치**: `Track2/LR/` (신규 트랙 전용 폴더; 기존 `Track1/LR/` 자산 보존)
**용도**: 태그 트랙 착수 전 전체 프로젝트 점검에서 확인한 사실·노선 변경 근거 기록 (실행 계획은 `docs/2/plan.md`)

---

## 0. 한 줄 요약

> 기존 "제목 스타일" 중심의 LR/RF 분석에서 드러난 한계(낮은 설명력 + 카테고리별 효과 이질성)를 디딤돌로 삼아, **카테고리별 핵심 태그가 조회수에 미치는 영향**을 분석하는 새 트랙을 추가한다. 기존 산출물은 폐기하지 않고 "베이스라인 → 한계 발견 → 새로운 접근"의 발표 서사로 통합한다.

---

## 1. 현재 상황 (Phase 1까지 확정 사항)

### 1-1. 완료된 작업

| Part | 상태 | 핵심 산출물 |
|---|---|---|
| **A — EDA** | ✅ 완료 (Phase 2) | `cleaned_USvideos.csv` (n=6249), 시각화 7종, H1/H2/H3 검증 |
| **B — Feature Engineering + LR** | ✅ 완료 (Phase 1+2) | `features_partB_v2.csv`, LR 결과, VIF/잔차 진단, VADER 감성 실험 |
| **C — Random Forest** | ✅ 완료 | `PartC_RF_final.ipynb`, `feature_importance_final.csv`, `model_comparison.csv` |

### 1-2. Phase 1까지의 핵심 발견

#### EDA (Part A)
- **H1 (카테고리 효과)**: Kruskal-Wallis p=6.1e-95, Gaming(median 1.43M) vs Nonprofits(37K) 38.7배 격차 → **카테고리는 가장 강한 구조적 예측자**
- **H2 (업로드 시간)**: ET 변환 후 hour p=4.5e-5, weekday p=8.3e-8 → Top 3 카테고리별 최적 시간대 상이
- **H3 (trending lag)**: Day 0(817K) > 8+days(127K), p=1.9e-28 → 빠른 트렌딩 = 높은 조회수 (상관관계, 인과 아님)

#### LR (Part B Phase 1)
- 전체 LR test R² = **0.116** (랜덤 split) — 메타데이터만으론 분산의 12%만 설명
- 시간 split R² = **−1.28** → temporal drift 발견 (모델 결함이 아닌 데이터셋 특성)
- **카테고리별 분리 회귀에서 부호 반전 발견**: `has_number`는 Music에서 −0.476* (유의), `exclaim_cnt`는 Music에서 음수, Entertainment에선 양수 → **H4 (효과의 카테고리별 이질성) 지지**

#### LR (Part B Phase 2 — 감성 추가 실험)
- VADER sentiment 3종(compound/pos/neg) 추가 → **ΔR² = −0.0022 (개선 없음)**
- 전체에선 무의미하지만 **Top 3 카테고리별로 부호 반전**: Howto & Style은 sent_pos +2.252* (긍정 제목 효과↑), Entertainment는 sent_pos −1.197* (긍정 제목 효과↓)
- → H4를 재확인하는 추가 증거, 그러나 **예측력 개선엔 실패**

#### RF (Part C)
- 최종 RF test R² = **0.245** (leakage 제외) — LR보다 나음, 그러나 여전히 낮음
- Feature Importance Top 5:
  1. trending_lag (0.149)
  2. **tag_cnt (0.141)** ⭐
  3. caps_ratio (0.135)
  4. title_len (0.099)
  5. publish_hour (0.086)

---

## 2. 왜 Phase 2로 스코프를 돌리는가

### 2-1. Phase 1의 두 가지 한계

**한계 1 — 낮은 설명력**
- 제목 객관 지표 + 시간 + 카테고리 + 감성까지 다 넣었는데 LR R² = 0.12, RF R² = 0.25
- 즉, "조회수 결정 요인의 75~88%는 우리가 본 변수들로 설명 안 됨"
- → 추가 정보원이 필요한 상황

**한계 2 — 효과의 카테고리별 이질성**
- 제목 효과 (Phase 1): `has_number` Music에선 음수, 다른 곳에선 무효
- 감성 효과 (Phase 2 실험): `sent_pos` Howto에선 양수, Entertainment에선 음수
- → **"모든 카테고리에 통하는 단일 공식"은 존재하지 않음**
- → 전체 풀링 분석은 본질적으로 한계가 있음

### 2-2. RF Feature Importance가 가리키는 다음 방향

RF Importance에서 **`tag_cnt`가 2위 (0.141)** 로 등장:
- 제목의 어떤 객관 지표(caps_ratio, title_len 등)보다 강함
- 그런데 우리는 태그를 **"개수"만** 사용했음 — 즉, "태그를 몇 개 달았는가"만 봤지 "어떤 태그인가"는 안 봄
- **"개수만 세도 2위인데, 내용까지 보면?"** → 자연스러운 다음 질문

### 2-3. 두 한계가 한 방향을 가리킨다

- 한계 1: 추가 정보원 필요 → **태그 내용**
- 한계 2: 카테고리별 분리 분석 필요 → **카테고리별 태그 분석**
- → **"카테고리별 핵심 태그가 조회수에 미치는 영향"** 이 두 한계에 동시 대응

### 2-4. 데이터 마이닝 과목 정합성

- 현재까지: 통계적 회귀 + 트리 모델 (이미 일반적)
- 추가: 텍스트 데이터(`tags`)에서 카테고리별 패턴 발굴 → **데이터 마이닝의 핵심 정신** (raw text → structured insight)
- 단순 회귀 프로젝트와의 차별화 요소

---

## 3. Phase 2에서 무엇을 할 것인가

### 3-1. 분석 대상 카테고리: Top 3 (median views 기준)

> **Gaming, Music, Film & Animation**

**선정 근거**:
- Part A의 H1·H2 분석에서 일관되게 사용된 기준
- 발표 슬라이드(A-4, A-5)와 시각화 5종이 이 기준으로 작성됨
- "흥행 카테고리에서 무엇이 흥행을 만드는가" narrative가 자연스러움

**Part B 기존 LR 분석(Entertainment/Music/Howto)과 다른 이유**:
- 기존 LR은 표본 수 기준 → 통계 검정 신뢰도 우선
- 태그 분석은 빈도 기반 + 평균 비교 → 표본 수 민감도 낮음
- 발표에서는 두 트랙을 **"방법론적 선택의 분리"** 로 명시:
  - 제목 효과 분석 → Entertainment/Music/Howto (표본 충분)
  - 태그 효과 분석 → Gaming/Music/Film & Animation (흥행 카테고리)
  - Music이 양쪽에 모두 포함 → 일관성 검증 지점

### 3-2. 분석 단계

```
Step 1. tags 컬럼 파싱
   - "tag1|"tag2"|"tag3"" 형식 → list of strings
   - 정규화: 소문자, 따옴표·공백 정리, 결측 처리([none] 등)
   - 영상별 태그 리스트 생성

Step 2. Top 3 카테고리별 인기 태그 추출
   - 각 카테고리 내 태그 빈도 집계
   - Top N (잠정 N=15~20) 추출
   - 카테고리 간 겹침 / 고유성 분석

Step 3. 태그별 조회수 효과 측정
   - 각 (카테고리, 태그) 쌍에 대해:
     · 태그 있는 영상의 median(log_views)
     · 태그 없는 영상의 median(log_views)
     · 차이 + Mann-Whitney U test (비모수, log-skew 데이터에 적합)
   - 효과 크기 + 통계적 유의성 동시 보고

Step 4. 시각화 & 표
   - 카테고리별 Top 태그 막대그래프 (효과 크기 기준 정렬)
   - 카테고리 × 태그 효과 히트맵 (공통 태그 비교용)
   - 발표 슬라이드용 핵심 표

Step 5. 모델 통합 (선택, 시간 허용 시)
   - Top 태그를 이진 feature로 변환
   - 기존 LR/RF에 추가 투입 → ΔR² 확인
   - 시간 부족 시 생략, Step 4 결과만으로 발표 진행
```

### 3-3. 명확히 안 하는 것

| 안 함 | 이유 |
|---|---|
| TF-IDF + SVD/PCA 차원 축소 | 시간 부족, 해석 가능성 우선 |
| LDA 토픽 모델링 | 시간 부족, 토픽 해석에 정성적 작업 필요 |
| Word2Vec / BERT 임베딩 | 수업 범위 초과, 정당화 부담 |
| 기존 LR/RF 결과 폐기 | 발표 서사의 "베이스라인"으로 보존 |
| 카테고리 4개 이상 확장 | 발표 분량·시간 제약 |

---

## 4. 발표 서사 통합 계획

### 4-1. 큰 흐름

```
[A] 문제 정의: 무엇이 조회수를 결정하는가?
       ↓
[B-1] EDA: 카테고리·시간·trending lag가 유의 (Part A 결과)
       ↓
[B-2] 1차 가설: 제목 스타일이 조회수를 좌우할 것이다
       ↓
[B-3] 1차 결과: LR R²=0.12, RF R²=0.25
       - 카테고리별 효과 이질성 발견 (H4 지지)
       - 그러나 전체 설명력은 낮음
       ↓
[B-4] Pivot: tag_cnt가 RF Importance 2위
       - "개수만으로도 강한데 내용까지 보면?"
       - 카테고리별로 분리해서 봐야 한다는 교훈 적용
       ↓
[B-5] 2차 분석: 카테고리별 핵심 태그 효과 ⭐ (Phase 2 신규)
       ↓
[C] 결론: 흥행 공식은 카테고리별로 다르고, 단일 메타데이터로는 부족하다
       - 한계 명시 + 향후 과제
```

### 4-2. Phase 2가 발표에서 차지할 분량

- 슬라이드 2~3장 (B-5 챕터)
- 발표 시간 약 2~3분 (전체 12분 중)
- 핵심 시각화 1~2개 (카테고리별 Top 태그 + 효과 비교)

---

## 5. 전체 프로젝트 점검 결과 (태그 트랙 착수 직전 기록)

> 태그 트랙 본격 착수 전, EDA·LR·RF 폴더 전체와 결과 CSV·스크립트를 재점검하며 확인한 사실과 노선 결정 근거를 기록한다. 실행 계획(Step 0~6, 확정된 설계 결정)은 별도 문서 `docs/2/plan.md`로 분리한다.

### 5-1. 검증된 사실 (태그 트랙 설계 근거)

- **median Top3 = Gaming / Music / Film & Animation — 코드 수준 확정**
  `Track1/EDA/EDA_Phase1/run_final.py`가 `views` median 내림차순 상위 3개를 그대로 사용하며, H1 그림 캡션도 "Gaming, Music, and Film & Animation lead"로 명시. §3-1의 카테고리 선택이 코드로 검증됨.
  (LR/감성 트랙 Top3 = Entertainment/Music/Howto는 **표본 수 기준** — `compare_sentiment.py`의 `value_counts().head(3)` 사용 확인. 두 집합의 분리는 의도된 방법론적 선택이며 Music이 교집합.)

- **EDA 전체가 비모수 검정 기반**
  H1 Kruskal-Wallis(H=490, p=6.1e-95), log-skew 분포. → 태그 효과 검정에 **Mann-Whitney U + median 차이**를 쓰는 것이 집안 방법론과 일관.

- **tag_cnt의 LR 계수 자체가 카테고리별로 이질적**
  `coef_comparison_top3_v2.csv`: Entertainment +0.019\*, Music +0.015\* (유의)이나 **Howto & Style +0.004 (무의미)**. → "태그 개수조차 카테고리별로 다르게 작동한다"가 내용 분석으로 넘어가는 직접적 다리.

- **감성(VADER) 추가의 예측 기여 ≈ 0**
  `model_compare_sentiment.csv`: M0(객관만) rand test R²=0.1188 → M1(+감성) 0.1166, ΔR²≈−0.002. 메타데이터 천장이 낮음을 재확인. → **태그 트랙의 가치는 R² 상승이 아니라 "어떤 태그가 흥행을 만드는가"의 해석**에 있다.

- **Step 5(모델 통합)용 평가 하니스가 이미 존재**
  `compare_sentiment.py` = M0 vs M1 → rand/time split R² + ΔR² + 카테고리별 계수 보고 구조. 태그 이진 feature 통합 시 그대로 재사용.

### 5-2. ⚠️ 발견된 cross-part 불일치 — trending_lag 누수 경계

- RF 최종 feature importance **1위 = trending_lag (0.149)**, tag_cnt는 2위(0.141). (`feature_importance_final.csv`)
- 그러나 Part B 누수 원칙은 trending_lag을 **사후(post-upload) 변수 → 제외** 대상으로 규정.
- 즉 **Part C 최종 RF(R²=0.245)는 Part B 기준으로는 누수를 포함**한 모델 (trending_lag 사용, days_on_trending만 제외).

**함의**:
1. trending_lag을 빼면 **tag_cnt가 업로드시점 feature 중 사실상 1위** → 태그 트랙 동기가 오히려 강화. 발표 프레이밍은 §2-2/§4-1의 "2위" 대신 **"업로드시점 변수 중 최강 예측자는 tag_cnt"**로 가는 것이 안전.
2. **Phase 4 머지 전 팀과 누수 경계 통일 필요.** R²=0.245 헤드라인이 trending_lag 포함값이라 누수 서사와 모순 가능.
3. `r2_ceiling_test.py` (S0 업로드시점만 / S1 +트렌딩메타 / S2 +참여도) 실행 시 trending_lag 제외 깨끗한 LR/RF/GBM R² 확보 가능 — 슬라이드용 권장.
4. **태그 트랙 자체는 trending_lag 미사용**으로 진행 (업로드시점 원칙 준수).

### 5-3. 부수 확인 사항

- `cleaned_USvideos.csv` 두 버전 존재: EDA_Phase1(17컬럼, Part B 피처 이전) / **EDA_Phase2(26컬럼, tags + 엔지니어링 피처 포함)**. 태그 트랙 입력은 Phase2 버전.
- 입력은 이미 n=6249로 중복 제거됨. 단 집안 패턴(`sort_values("views").drop_duplicates("video_id", keep="last")`)에 맞춰 방어적 재중복제거 권장.
- 카테고리별 n(특히 Gaming, Film & Animation)은 Step 0에서 실측 확인 필요. Gaming이 빈도 임계(≥5) 적용 후 태그가 너무 적으면 "Music + 1개"로 축소하는 fallback 유지.

---

**다음 문서**: `docs/2/plan.md` — 태그 트랙 실행 계획 (Step 0~6, 확정된 설계 결정)
