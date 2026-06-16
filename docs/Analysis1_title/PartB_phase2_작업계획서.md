# Part B — Phase 2 작업계획서 (Feature Engineering & LR 개선)

**담당**: Donghyun (Part B — LR)
**가설**: H4 — 제목 스타일이 조회수에 영향을 주며, 그 효과는 카테고리마다 다르다
**Phase 2 목표**: Phase 1 베이스라인의 두 한계(① UTC 시간 어긋남, ② 객관 지표만의 낮은 설명력)를 보완하고, "베이스라인 → 한계 발견 → 개선" 발표 서사를 완성한다.
**입력**: `EDA/EDA_Phase2/cleaned_USvideos.csv` (Part A 인계, ET 시간 + 제목변수 8종 내장, n=6249)

---

## 0. Phase 1 인수인계 (확정 사항)

| 항목 | 확정 내용 |
|---|---|
| 확정 feature set | title_len, caps_ratio, exclaim_cnt, question_cnt, has_number, has_bracket, tag_cnt (**word_count 제외** — title_len과 VIF 7.77) |
| 타깃 | `log_views` (log1p) |
| 시간 변수 | hour_bin(6시간 4구간 원-핫, ref=18-23) + publish_weekday(범주형) |
| 모델 | statsmodels OLS (통계 해석 목적) |
| 성능 보고 규칙 | **랜덤 split = 베이스라인 / 시간 split = drift 정도**, 반드시 병기 |
| 누수 제외 | likes, comment_count, dislikes, days_on_trending, trending_lag, log_* |
| 발견(finding) | 시간 split R²<0 = temporal drift (모델 결함 아님) |

---

## 1단계 — ET 시간 변환 반영 ✅ **완료** (2026-06-17)

Part A가 Phase 2 정제본에서 `publish_hour`/`publish_weekday`를 **UTC→ET(DST 자동처리)**로 변환. Part B는 입력 경로 교체 후 노트북 4개 전부 재실행.

**결과 요약** (상세: `phase1_작업일지.md §6`):
- UTC→ET로 hour_bin 라벨 76.7%, 요일 13.5% 이동
- 제목변수 계수·VIF·잔차·성능(랜덤 +0.119 / 시간 −1.277) **사실상 불변** → 결론 견고
- **변화 1건**: 요일 변수가 ET에서 **주말 유의** (Sat +0.239\*, Sun +0.266\*). UTC에선 전부 무의미 → H2(시간 효과)가 LR에서도 재현
- drift는 ET로도 유지 → 시간대 편향만으로 설명 안 됨(수집기간 편중 잔존)

**산출**: `features_partB.csv`(ET), `coef_comparison_top3_v2.csv`(ET) 재생성 완료.

---

## 2단계 — VADER 감성 파생변수 추가 (Phase 2 핵심) ✅ **완료** (2026-06-17)

**결과 요약** (스크립트: `LR/make_sentiment.py`, `LR/compare_sentiment.py`):

| 지표 | M0 (객관만) | M1 (+감성) | 판정 |
|---|---|---|---|
| 랜덤 split test R² | 0.1188 | 0.1166 | **ΔR² = −0.0022 (개선 없음)** |
| train R² | 0.1186 | 0.1210 | 소폭↑ (과적합 신호) |
| 시간 split test R² | −1.271 | −1.271 | 불변 |

- **전체 감성 계수 전부 무의미**: sent_compound p=0.59, sent_pos p=0.77, sent_neg p=0.053(경계)
- **"객관 지표가 이미 커버"는 아님**: VADER ↔ caps_ratio(r=0.03)·exclaim_cnt(r=0.02) 거의 무관. 즉 객관 지표가 감성을 흡수한 게 아니라, **제목 감성 자체가 조회수와 약하게만 연결**(제목 50%가 중립 compound=0).
- **H4 관점에선 강한 추가 증거** — Top3 카테고리별 감성 효과가 **부호 반대로 갈림**:

| feature | Entertainment | Music | Howto & Style |
|---|---|---|---|
| sent_compound | 0.621 | −0.365 | **−2.037\*** |
| sent_pos | **−1.197\*** | 1.189 | **+2.252\*** |
| sent_neg | 1.231 | 0.030 | **−2.122\*** |

  → Howto&Style: 긍정 제목→조회수↑ / Entertainment: 긍정 제목→조회수↓. 전체로는 상쇄되어 무의미하지만 **카테고리별로는 효과 방향이 반대** = H4 강한 지지.

**판정 적용**: ΔR²≈0 → **"객관 지표만으론 설명력 약함"이 감성 추가로도 유지** = 발표의 "한계" 강화. 동시에 **카테고리별 감성 부호 반전**은 "개선" 서사가 아닌 **H4 심화 증거**로 활용. 단순 예측 성능이 아닌 **해석적 발견**으로 포지셔닝.

**산출**: `features_partB_v2.csv`(감성 포함), `model_compare_sentiment.csv`.

---

### (원 계획) VADER 감성 파생변수 추가

**문제의식**: Phase 1은 "객관 지표(대문자·느낌표·괄호)"만으로 자극성을 측정 → 설명력 낮음(R²≈0.12). 제목의 **감정 톤(긍/부정 강도)** 이라는 직접 신호를 추가하면 개선되는지 검증.

### 2-1. 피처 생성
- 도구: `nltk` VADER (`SentimentIntensityAnalyzer`) — 짧은 텍스트·SNS 톤에 적합, 사전 기반이라 학습 불필요
- 제목(`title`)에서 추출:

| feature | 정의 | 자극성 해석 |
|---|---|---|
| `sent_compound` | VADER compound (−1~+1) | 전반적 감정 강도·방향 |
| `sent_pos` | positive 비율 | 긍정 자극 |
| `sent_neg` | negative 비율 | 부정/충격 자극 |
| (옵션) `sent_abs` | `abs(compound)` | 중립 대비 "감정적임" 자체 |

> 누수 점검: 제목은 업로드 시점 확정 → 누수 없음. ✅

### 2-2. 모델 비교 (개선 여부 정량 측정)
- **M0 (Phase 1 확정)**: 객관 제목변수 + 시간 + 카테고리
- **M1 (+감성)**: M0 + VADER 변수
- 비교 지표: **랜덤 split R²/RMSE**(베이스라인 기준) + 계수·p-value + ΔR²(M1−M0)
- 시간 split도 병기(drift 영향 점검)

### 2-3. 판정 규칙 (핸드오프 명시)
- **개선 유의(ΔR² 의미 있음)** → "객관 지표만으론 약하다 → 감성 신호로 개선" **개선 서사**에 포함
- **개선 미미** → "caps_ratio·exclaim_cnt가 이미 감성 신호를 커버한다"는 **결론**으로 사용 (VADER와 객관 지표 상관 점검 첨부)
- 어느 쪽이든 발표에 쓸 결과 → **둘 다 유효한 산출**

### 2-4. H4 재확인
- Top3 카테고리(Entertainment/Music/Howto&Style)별로 감성 변수 계수 방향·유의성 비교
- 예상 포인트: Music에서 부호 반전 패턴(has_number·exclaim)이 감성 변수에서도 나타나는지

**산출**: `features_partB_v2.csv`(감성 포함), `model_compare_sentiment.csv`(M0 vs M1), 갱신 노트북.

---

## 3단계 — (옵션) 정밀화

1. **카테고리 × 제목변수 상호작용항** — H4를 단일 모델 교차항으로도 검정 (분리 회귀의 보완 증거)
2. **Robust SE (HC3)** — 시간 split·이분산(BP 기각) 대응, 계수 추론 엄밀화
3. **VADER vs 객관 지표 상관 분석** — 2-3 "이미 커버" 결론의 근거 도표

> 3단계는 2단계 결과에 따라 선택. 발표 시간·분량에 맞춰 취사.

---

## 4단계 — 종합 & 발표 연결

발표 스토리(Part A 핸드오프 §6 골격과 정합):

| 단계 | 메시지 | Part B 근거 |
|---|---|---|
| 베이스라인 | 제목/시간/카테고리만으로 예측 | 랜덤 R²≈0.12 |
| 한계 발견 | drift(시간 R²<0) + 낮은 설명력 | 시간 split −1.28, UTC 어긋남 |
| 개선 | ET 변환(주말 유의) + 감성 피처 | 1·2단계 결과 |
| 결론 | 자극성 효과는 카테고리 의존(H4), 조회수는 참여도가 지배 | 계수 비교 + 누수 실험 |

---

## 데이터 / 파일 경로

| 항목 | 경로 |
|---|---|
| 입력 (Phase2 정제본, ET) | `EDA/EDA_Phase2/cleaned_USvideos.csv` |
| publish_time (시간 split 키) | `EDA/EDA_Phase1/USvideos.csv` |
| 작업 폴더 | `LR/` |
| 출력 — feature 테이블 | `LR/features_partB.csv` → (감성) `features_partB_v2.csv` |
| 출력 — 모델 비교 | `LR/model_compare_sentiment.csv` |
| 출력 — 시각화 | `LR/fig_lr_*.png` |

---

## 진행 순서

```
1. ET 변환 반영 ✅ 완료
   ↓
2. VADER 감성 피처 추가 → M0 vs M1 비교 ✅ 완료 (ΔR²≈0, H4 카테고리별 부호 반전 발견)
   ↓
3. (옵션) 상호작용항 / robust SE / 상관 분석
   ↓
4. 종합 + 발표 스토리 확정
```

**다음 착수**: 3단계(옵션) 또는 4단계(종합). 2단계의 "카테고리별 감성 부호 반전"을 단일 모델 **category×감성 상호작용항**으로 정식 검정하면 H4 증거가 더 단단해짐.
