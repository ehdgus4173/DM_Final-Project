# Part B 작업계획서 — Feature Engineering & Linear Regression

**담당**: Donghyun (Part B)
**산출물**: feature 테이블 + LR 모델 결과 + 통계 해석
**검증 가설**: H4 — "제목 스타일(객관 지표)이 조회수에 영향을 주며, 그 효과는 카테고리마다 다르다"

---

## 0. 확정된 설계 전제

| 항목 | 결정 | 근거 |
|---|---|---|
| 타깃 변수 | `log_views` (log1p) | A가 이미 생성, 분포 왜도 보정 |
| Feature 범위 | category + publish 시간 + **객관적 제목 파생변수만** | 누수(leakage) 방지 |
| 제외 변수 | likes, comment_count, dislikes, days_on_trending, trending_lag | 업로드 시점에 알 수 없음 (사후 측정치) |
| "자극성" 측정 | **객관 지표만** (감성 지표는 보류) | 1차는 깨끗한 베이스라인 확보 우선 |
| 시간 기준 | **UTC 유지** | A의 EDA·해석과 일관성 유지 |
| 시간 인코딩 | **구간(6시간 4구간) 원-핫** | 비선형 가정 없음, 계수·p-value 해석 명확 |
| 카테고리 차이 | **그룹별 분리 회귀** (교차항 X) | 해석·발표 직관성 |
| 모델 | Linear Regression (statsmodels OLS) | 베이스라인, 통계 검증 목적 |

> **누수 관련 메모**: A가 만든 `cleaned_USvideos.csv`에는 사후 변수(likes 등)가 포함돼 있으나, Part B에서는 사용하지 않는다. category·publish 시간·title·tags만 사용.

---

## 1단계 — Feature Engineering (객관적 제목 파생변수)

`title`, `tags` 컬럼에서 추출 (전부 업로드 시점 = 누수 없음)

| feature | 정의 | "자극성" 해석 |
|---|---|---|
| `title_len` | 제목 글자 수 | 길이 |
| `word_count` | 단어 수 | 정보량 |
| `caps_ratio` | 대문자 비율 | 강조 / 외침 |
| `exclaim_cnt` | `!` 개수 | 감정 강조 |
| `question_cnt` | `?` 개수 | 호기심 유발 |
| `has_number` | 숫자 포함 여부 (0/1) | 리스트형 클릭베이트 |
| `has_bracket` | `[]` · `()` 포함 여부 (0/1) | 태그형 제목 |
| `tag_cnt` | 태그 개수 | 노출 최적화 |

**시간 변수**: A가 만든 `publish_hour`(UTC), `publish_weekday` 그대로 사용

`publish_hour`는 **구간 4개로 묶어 원-핫 인코딩** (UTC 기준, 라벨 중립):

| 구간 | 시간 (UTC) |
|---|---|
| `hour_0_5` | 0–5시 |
| `hour_6_11` | 6–11시 |
| `hour_12_17` | 12–17시 |
| `hour_18_23` | 18–23시 (기준 그룹으로 drop 가능) |

> 선정 이유: 원시값은 시간을 선형으로 강제(23시↔0시 인접성 무시), 순환 인코딩(sin·cos)은 계수 해석 불가. 구간은 "어느 시간대가 유의하게 높은가"를 p-value로 바로 검정 가능 → 통계 해석 목적의 Part B에 최적. (순환 인코딩은 4단계/C파트 RF에서 시도)

**산출**: feature 테이블 (CSV) → `LR/features_partB.csv`

---

## 2단계 — Baseline LR (전체 데이터) + 통계 검증

**모델식**
```
log_views ~ 제목파생변수 + 시간변수 + category(원-핫)
```
- 전체 데이터를 **단일 모델**로 적합
- 도구: `statsmodels` OLS (계수·p-value·신뢰구간 일괄 출력)

**통계 검증 4종 세트** (Part B 핵심 산출물)

1. **계수 + p-value** — 어떤 제목 변수가 유의한가, 방향은?
2. **VIF** — 다중공선성 점검 (`title_len` ↔ `word_count` 중복 의심 → 필요시 하나 제거)
3. **잔차 분석** — 잔차 vs 예측값, Q-Q plot, 등분산성
4. **R² / RMSE** — **시간 기반 train/test split** 에서 측정
   - 통계 검증(계수·VIF·잔차)은 **train**
   - 성능 지표(R²·RMSE)는 **test**

**산출**: LR 결과 요약표 + 잔차/Q-Q 시각화 + 통계 해석 노트

---

## 3단계 — 카테고리별 분리 회귀 (H4 후반부)

- Top3 카테고리(Gaming / Music / Film & Animation) **각각 별도 LR 적합**
- "튜닝"이 아니라 **그룹별 재적합(re-fit) + 계수 비교**
- 비교 포인트: 같은 제목 변수의 **계수 방향·크기·유의성이 그룹마다 다른가**
  - 예: `exclaim_cnt`가 Gaming에선 +, News에선 −일 수 있음 = 상호작용 증거

**산출**: 카테고리별 계수 비교표 + 해석 ("자극성의 효과는 카테고리에 따라 다르다" 검증)

---

## 4단계 — 피드백 루프 (다음 반복)

1차 결과 해석 후 다음 개선 항목으로 진행:

1. **감성 파생변수 추가** — VADER 감성점수, 자극 키워드 사전 매칭
   → "객관 지표만으론 설명력이 약하다 → 감성 신호로 개선" 스토리
2. **ET 시간 변환** — 발견된 구멍 정식화
   → "데이터셋이 UTC 기준이라 미국 시청자 체감 시간과 어긋남을 발견 → ET 변환으로 재분석"
   → 발표 스토리: "1차 분석에서 한계 발견 → 개선" 흐름

---

## 데이터 / 파일 경로

| 항목 | 경로 |
|---|---|
| 입력 (정제 데이터) | `EDA/cleaned_USvideos.csv` |
| 카테고리 매핑 | `EDA/US_category_id.json` |
| 작업 폴더 | `LR/` |
| 출력 — feature 테이블 | `LR/features_partB.csv` |
| 출력 — LR 코드 | `LR/run_lr.py` |
| 출력 — 결과 시각화 | `LR/fig_lr_*.png` |

---

## 진행 순서 요약

```
1. Feature Engineering (객관 제목 파생)  → features_partB.csv
   ↓
2. Baseline LR (전체) + 통계검증 4종      → 계수·VIF·잔차·R²/RMSE
   ↓
3. 카테고리별 분리 회귀 (Top3)            → 계수 비교표
   ↓
4. 피드백 루프 (감성 변수 + ET 변환)
```

**모든 설계 결정 완료** → 1·2단계 코드 착수 가능
