# Phase 1 — 태그 분석 (Track 2) 결과 기록

> 이번 세션에서 수행한 Track 2 "태그 내용 효과" 분석의 전체 기록.
> 설계 결정·실행 단계·결과·해석·한계를 한 문서로 정리한 마스터 레코드.
> (세부 계획은 `plan.md`, 진행 로그는 `worklog.md` 참조)

---

## 0. 배경 & 목적

- **상위 트랙(Part B, Track1)**: 제목 스타일(objective metrics) → log_views 선형회귀.
- **RF feature importance**에서 `tag_cnt`가 상위(업로드 시점 변수 중 사실상 1위; `trending_lag`은 post-upload 누수라 제외 시). → "태그의 **개수**가 아니라 **내용(어떤 태그)**이 조회수에 영향을 주는가?"로 확장.
- **핵심 질문**: 태그 내용이 조회수에 영향을 주며, 그 효과가 **카테고리별로 다른가** (H4 이질성).
- **누수 차단(leakage discipline)**: 업로드 시점 메타데이터만 사용. 타깃 = `log_views`. 참여 지표(likes/comments)·post-upload 변수 전면 배제 — 비협상 설계 원칙.

## 1. 데이터

- 입력: `Track1/EDA/EDA_Phase2/cleaned_USvideos.csv` (n=6249, 팀원 EDA 산출물)
- **중앙값(median log_views) 기준 TOP3 카테고리 = Gaming / Music / Film & Animation**
  - 표본 수 기준 TOP3(Entertainment/Music/Howto, 제목 LR 트랙이 사용)와 **다름** — Music만 겹침. 의도적 분리.

## 2. 설계 결정 (확정)

| # | 항목 | 결정 |
|---|------|------|
| 1 | 대상 카테고리 | Gaming / Music / Film & Animation (median TOP3) |
| 2 | 카테고리별 Top N | 15 |
| 3 | 빈도 임계 | 카테고리 내 document-frequency ≥ 5 |
| 4 | 비교 방식 | 카테고리 내(in-category) with vs without tag |
| 5 | 효과 측정 | median(log_views) 차이 + 양측 Mann-Whitney U + rank-biserial r |
| 6 | 해시태그 | 선행 `#` 제거 |
| 7 | 다중비교 보정 | raw p(`*`, α=.05) + BH-FDR(q, `sig_fdr`) |
| 8 | 모델 통합(Step 5) | 선택 사항 → **Phase 2로 이관** |

추가 결정: 불용어 `the` 제거 / Gaming은 끝까지 추적 후 무의미하면 한계 사례로 유지.

## 3. 실행 단계 & 산출물

모든 노트북·CSV·PNG는 현재 **`Track2/phase1/`** 에 위치 (세션 중 `Track2/LR/`에서 이동). 상대경로 깊이가 같아 입력 로드 정상.

| Step | 노트북 | 내용 | 산출 |
|------|--------|------|------|
| 0 | `step0_tag_analysis.ipynb` | 로드·중복 제거, median TOP3 확인 | (검증) |
| 1 | `step1_tag_analysis.ipynb` | 태그 파싱·정규화 | `step1_tags_parsed.csv` |
| 2 | `step2_tag_analysis.ipynb` | 카테고리별 top 태그(≥5, top15) | `step2_top_tags.csv` (45행) |
| 3 | `step3_tag_analysis.ipynb` | 효과 + 유의성 검정 | `step3_tag_effect_top3.csv` |
| 4 | `step4_tag_analysis.ipynb` | 3패널 시각화 + 발표 요약 | `step4_fig_tag_effects.png` |

### 단계별 확인값
- **Step 0**: 6249→6249 (중복 없음). 카테고리 n: Gaming **101**, Music **792**, Film & Animation **306**.
- **Step 1**: `step1_tags_parsed.csv` 컬럼 = video_id·category·log_views·tags_joined. `n_tags == tag_cnt` 일치율 1.000. 무태그 영상 **244개(3.9%)**, 평균 ~20 태그/영상.
- **Step 2**: 임계 통과 태그 수 — Gaming 32 / Music 338 / Film 130. **카테고리 vocabulary 거의 분리**(Music∩Film에 `official`만 공통, 3-way 공통 0) → 공통-태그 히트맵 폐기, 3패널 비교로 대체. 노이즈: Film의 ScreenJunkies 채널 유사 중복(honest trailers/trailer 등).
- **Step 3**: `step3_tag_effect_top3.csv` 컬럼 = category, tag, n_with, n_without, median_with, median_without, diff, U, p, sig, q, sig_fdr, rank_biserial. **FDR 생존: Gaming 0 / Music 4 / Film 3**.
- **Step 4**: 3패널 barh(파랑=FDR q<.05 / 초록=raw p<.05 / 회색=n.s.), `sharex`로 크기 비교 가능.

## 4. 핵심 결과 (FDR 유의 태그)

| category | tag | diff(log) | q |
|----------|-----|-----------|------|
| Film & Animation | disney | **+1.457** (≈4.3배) | 0.0027 |
| Film & Animation | trailer | +0.928 | 0.0041 |
| Film & Animation | movie | +0.987 | 0.0179 |
| Music | pop | +0.591 | 0.0415 |
| Music | country | −0.785 | 0.0115 |
| Music | music | −0.895 | 0.0017 |
| Music | alternative | −1.157 | 0.0177 |

- **Film & Animation**: 흥행(+) disney·movie·trailer / 역효과(−) 없음.
- **Music**: 흥행(+) pop / 역효과(−) alternative·music·country.
- **Gaming**: FDR 생존 0. raw 유의는 `games`(−1.95, n=8 노이즈)뿐 → **한계 사례 확정**.

## 5. 해석 (발표 서사)

1. **카테고리별 흥행 태그가 다름** → H4(이질성) 지지. 단일 "좋은 태그"는 없고 맥락 의존.
2. **일반적/포괄적 태그일수록 조회수가 낮은 경향**(music, alternative, country, games) → 선택 효과 해석: 히트작은 흔한 태그를 덜 단다(또는 흔한 태그 영상이 평범).
3. **`disney` 효과는 태그 메커니즘이 아니라 브랜드/IP 교란** → 상관 ≠ 인과. 발표에서 명시.
4. **rank-biserial 대부분 |r|<0.3** (분포 겹침 큼) → 유의성은 부분적으로 큰 표본수에 기인. "통계적 유의 ≠ 실무적 큰 효과" 톤 유지.

## 6. 한계

- Gaming 소표본(n=101) → 태그 효과 미검출.
- 태그 이진은 희소(sparse), 유의성 일부 n-driven.
- `disney` = IP 교란.
- **단변량(univariate)**: 각 태그를 독립 검정, 다른 변수 통제 없음 → **Phase 2(다변량 LR 통합)로 연결**.

## 7. 다음 (Phase 2로 이관)

- Step 5 LR 재설계: pooled M0 vs M_tags(FDR-7) — **확정**.
- 과제(Assignment01/02) 검토로 새로 도입할 방법론(교차검증·왜도 로그변환·permutation importance·Lasso 등) → `docs/2/phase2/plan.md` 참조.
