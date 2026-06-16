# Phase 2 — Tag Track: 상세 작업계획서 (Execution Plan)

**담당**: Donghyun (Part B 확장)
**작성**: 2026-06-17, 발표 D-1
**전제 문서**: `docs/2/Background.md` (노선 변경 근거·점검 기록)
**입력**: `Track1/EDA/EDA_Phase2/cleaned_USvideos.csv` (n=6249, 26컬럼, tags·log_views·category·tag_cnt 포함)
**작업 노트북**: `Track2/LR/step{N}_tag_analysis.ipynb` (Step별 분리, 각 self-contained — 로드/전처리 중복은 구조화 tradeoff)
**환경**: Anaconda `dm`, Python 3.12.13, VS Code

> **용어**: "Phase"는 프로젝트 전체 단위(제목 트랙=Phase 1, 태그 트랙=Phase 2). "Step"은 이번 태그 트랙 내부의 실행 단계(0~6).
> **폴더 구조**: `Track1/`(EDA·LR·RF = 기존 제목 트랙) / `Track2/`(LR = 태그 트랙) / `docs/`. 노트북은 `Track2/LR/`에 위치하므로 입력 상대경로는 `../../Track1/EDA/...`.

---

## A. 누수(leakage) 스코프

태그 트랙은 **업로드시점에 알 수 있는 정보만** 사용한다.

- **사용**: `tags`, `category`, `log_views`(타깃)
- **미사용**: `trending_lag`, `days_on_trending`(사후), `likes`/`dislikes`/`comment_count`(참여도)
- **trending_lag**: RF importance 1위였으나 Part B 기준 사후 변수 → **본 트랙에서 사용 안 함**. (근거·함의: Background §5-2)

---

## B. 확정된 설계 결정

| # | 항목 | 확정값 | 근거 |
|---|---|---|---|
| 1 | 분석 카테고리 | **Gaming / Music / Film & Animation** | median views Top3 (run_final.py 검증) |
| 2 | Top N | **15** (임계 통과 후 상위 15) | 소표본 카테고리는 20 무의미 |
| 3 | 빈도 임계 | **카테고리 내 document-frequency ≥ 5** | Gaming 소표본 노이즈 제거 |
| 4 | 비교군 | **in-category** (같은 카테고리 내 보유 vs 미보유) | 전 분석이 카테고리별; 카테고리 효과 혼입 방지 |
| 5 | 효과지표 | **median(log_views) 차이 + 양측 Mann-Whitney U** (+ rank-biserial r 보조) | EDA 비모수 일관, log-skew |
| 6 | 해시태그 | **선두 `#` 제거 후 정규화** | `#x`와 `x` 중복 병합 (데이터에서 확인) |
| 7 | 다중비교 | **raw p(`*`, α=0.05) + BH-FDR 보정(`q`, sig_fdr) 병기** | ~45개 검정, 저비용 robustness |
| 8 | Step 5(모델통합) | **선택**, 하면 `compare_sentiment.py` 하니스 재사용 | 감성 전례상 ΔR² 미미 → 해석 프레임 |

---

## Step 0 — 셋업 & 데이터 검증  ✅ 완료 (2026-06-17)

- **작업**
  - `Track2/LR/step0_tag_analysis.ipynb` 생성, imports: `pandas`, `numpy`, `scipy.stats.mannwhitneyu`, `statsmodels.stats.multitest.multipletests`, `matplotlib`/`seaborn`, `collections.Counter`
  - 로드: `../../Track1/EDA/EDA_Phase2/cleaned_USvideos.csv`
  - 방어적 재중복제거: `sort_values("views").drop_duplicates("video_id", keep="last")`
  - 컬럼 존재 확인(category·log_views·tags·tag_cnt)
  - `category` value_counts + median(views) 내림차순 출력 → **3개 카테고리 n 실측, median Top3 재확인**
- **산출**: 데이터 로드 셀 + 카테고리 분포 표
- **결과**: dedup 6249→6249(중복 없음), shape (6249, 26), 컬럼 OK. **median Top3 = Gaming/Music/Film & Animation (match=True)**. 타깃 n: **Gaming 101 / Music 792 / Film & Animation 306**.
- **판단**: Gaming n=101로 소표본이나 임계(≥5) 적용 시 진행 가능 → 3개 카테고리 유지. Step 2~3에서 Gaming 유의 태그 <3이면 §E fallback(3→2) 재검토.

## Step 1 — tags 파싱·정규화

- **작업**
  - `|` 분리 → 토큰 양끝 `"` 제거 → strip → lowercase → 선두 `#` 제거 → 빈 토큰 제거
  - 결측(`[none]`, NaN, 빈값) → 빈 리스트(= "태그 없음" 그룹)
  - `tag_list` 컬럼 생성
  - 정합성: `len(tag_list)` vs `tag_cnt` 분포 비교 (완전 일치 불요, 큰 괴리만 점검)
- **산출**: `Track2/LR/step1_tags_parsed.csv` (video_id·category·log_views·tags_joined; Step 2+ 입력) + 파싱 통계(결측 영상 수, 평균 태그 수)
- **주의**: `#` 제거로 해시태그/일반형 병합됨을 worklog에 기록
- **완료 기준**: 파싱 실패·결측 비율 보고 + 샘플 수동 검수

## Step 2 — 카테고리별 인기 태그 추출

- **작업**
  - 각 카테고리 subset에서 태그 **document-frequency** 집계(영상당 `set`으로 중복 제거)
  - 임계 ≥5 필터 → 빈도순 정렬 → **Top 15**
  - 카테고리 간 겹침/고유 태그 분석(set 연산)
- **산출**: 카테고리별 Top 태그 빈도 표 + 겹침 요약
- **주의**: term-frequency 아닌 document-frequency(몇 개 영상에 등장) 기준
- **완료 기준**: 각 카테고리 임계 통과 태그 수 보고 (Gaming 부족 시 fallback 신호)

## Step 3 — 태그별 조회수 효과 + 유의성

- **작업** — 각 (카테고리, 태그)에 대해:
  - groupA = 카테고리 내 태그 **보유** 영상의 `log_views`, groupB = **미보유** 영상의 `log_views`
  - `median_with`, `median_without`, `diff = median_with − median_without`
  - `mannwhitneyu(A, B, alternative="two-sided")` → U, p
  - rank-biserial r (보조 효과크기)
  - 유효조건: `n_with ≥ 5`
  - 전체 (cat,tag) p 모아 **BH-FDR**(`multipletests(method="fdr_bh")`) → `q`, `sig_fdr`
- **산출**: `Track2/LR/step3_tag_effect_top3.csv`
  컬럼: `category, tag, n_with, n_without, median_with, median_without, diff, U, p, sig, q, sig_fdr, rank_biserial`
- **주의**: `diff`는 자연로그 단위. 발표용 배수 효과 = `exp(diff)` (예: diff≈0.69 → 조회수 ≈2배)
- **완료 기준**: CSV 생성 + 카테고리별 유의(raw/fdr) 태그 수 요약

## Step 4 — 시각화 & 표

- **작업**
  - 카테고리별 Top 태그 막대그래프: x = `diff`(효과크기), 정렬, 유의 강조(색/별표), 카테고리별 패널
  - 카테고리 × 공통 태그 효과 히트맵(겹치는 태그의 카테고리별 `diff` → 이질성 가시화). 공통 태그 적으면 생략하고 카테고리별 Top diff 비교 패널로 대체
  - 발표 요약표: 카테고리별 상위 흥행/역효과 태그 3개씩
  - 스타일: `run_final.py` 팔레트 일관(PRIMARY `#3B4FE4`, ACCENT `#1A7F5A`, GRAY `#C8CDD6`), dpi=200
- **산출**: `Track2/LR/step4_fig_tag_effects.png` (+ 조건부 `Track2/LR/step4_fig_tag_heatmap.png`)
- **완료 기준**: 카테고리 간 효과 차이를 보여주는 그림 ≥1

## Step 5 — (선택) 모델 통합

- **게이트**: Step 1~4 완료 + 시간 여유. 효과 약하면 생략하고 Step 4로 발표.
- **작업**
  - 선정 Top 태그(유의 태그 union)를 이진 feature로 변환
  - `Track1/LR/features_partB_v2.csv`에 `video_id`로 조인(키 존재 확인 필수)
  - `compare_sentiment.py` 구조 복제: **M0(객관만) vs M_tags(+태그 binary)** → rand/time split R² + ΔR² + 카테고리별 계수
- **산출**: `Track2/LR/step5_model_compare_tags.csv`
- **주의**: 감성 ΔR²≈−0.002 전례 → R² 상승 기대 낮음. **null이어도 "메타데이터 천장 낮음" 서사로 보고**. 프레임 = "내용이 개수(tag_cnt) 너머 신호를 더하는가"
- **완료 기준(선택)**: ΔR² 보고 + 해석

## Step 6 — 문서화 & 발표 반영

- **작업**
  - `docs/2/worklog.md` 작성: 진행 중 결정/발견/보정(실제 임계 통과 수, fallback 발동 여부, 주요 흥행 태그, Step 5 결과)
  - 발표 **B-5 챕터 2~3장**: ① 왜 태그(RF importance + 카테고리 이질성), ② 카테고리별 Top 태그 효과 그림, ③ 핵심 인사이트 + 한계
  - "왜 태그로 갔는가" 정당화를 **trending_lag 재프레이밍**(업로드시점 최강 예측자 = tag_cnt)으로 보강
- **완료 기준**: worklog 작성 + 슬라이드 반영

---

## C. 파일 / 경로

| 항목 | 경로 |
|---|---|
| 입력 | `Track1/EDA/EDA_Phase2/cleaned_USvideos.csv` |
| 작업 노트북 | `Track2/LR/step{N}_tag_analysis.ipynb` (Step별) |
| 중간 산출(파싱) | `Track2/LR/step1_tags_parsed.csv` |
| 효과 표 | `Track2/LR/step3_tag_effect_top3.csv` |
| 시각화 | `Track2/LR/step4_fig_tag_*.png` |
| (선택) 모델 비교 | `Track2/LR/step5_model_compare_tags.csv` |
| (참고) 기존 자산 | `Track1/LR/` (features_partB_v2.csv, compare_sentiment.py 등), `Track1/RF/`, `Track1/EDA/` |
| 기록 | `docs/2/Background.md`(점검 기록) · `docs/2/plan.md`(본 문서) · `docs/2/worklog.md`(진행 일지) |

## D. 일정 (D-1)

| Step | 내용 | 소요 | 우선순위 |
|---|---|---|---|
| 0 | 셋업·검증 | 20분 | 필수 ✅ |
| 1 | tags 파싱 | 30분 | 필수 |
| 2 | 카테고리별 Top 태그 | 30분 | 필수 |
| 3 | 효과 측정 + 유의성 | 1시간 | 필수 |
| 4 | 시각화 + 표 | 1시간 | 필수 |
| 5 | 모델 통합 | 1~2시간 | 선택 |
| 6 | 문서·슬라이드 반영 | 1시간 | 필수 |

**총 예상**: ~4.5시간(Step 5 제외), ~6.5시간(포함)

## E. 위험 & 대응

| 위험 | 대응 |
|---|---|
| tags 파싱 실패/결측 다수 | `[none]`·빈값 표준 처리, 결측 영상 별도 그룹 |
| Gaming 소표본(n=101) → 임계 후 태그 부족 | fallback: 카테고리 3→2 축소(Music + Film & Animation), 또는 차순위 median 카테고리(Comedy) 대체 |
| 효과 약해 인사이트 빈약 | Step 4까지 마치고 Step 5 진행 여부 판단 |
| 다중비교로 거짓양성 우려 | FDR `q` 병기로 대응 |
| 발표 시간 초과 | 카테고리 3→2, 히트맵 생략 |

## F. 성공 기준

- ✅ 3개(불가 시 fallback 2개) 카테고리 각각에서 유의 흥행 태그 식별(가능한 범위 — Gaming은 소표본 한계 명시)
- ✅ 카테고리 간 효과 차이 시각화 ≥1
- ✅ "왜 태그로 갔는가"가 데이터 근거(RF importance, 카테고리 이질성, tag_cnt 업로드시점 1위)로 정당화
- ✅ 기존 LR/RF 폐기 없이 "베이스라인 → 발견 → 확장" 서사로 통합

---

**다음 문서**: `docs/2/worklog.md` — Step 진행 중 결정·발견·보정 기록 (Step 1 착수 시 생성)
