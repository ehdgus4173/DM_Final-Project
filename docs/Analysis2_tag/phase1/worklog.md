# Phase 2 — Tag Track: 진행 일지 (worklog)

**기간**: 2026-06-17 (발표 D-1)
**범위**: Step 0 ~ Step 3 완료 시점 기록
**관련 문서**: 계획 `docs/2/plan.md` · 점검 기록 `docs/2/Background.md`
**작업 위치**: `Track2/LR/` (Step별 self-contained 노트북)

---

## 진행 현황 요약

| Step | 내용 | 상태 | 산출물 |
|---|---|---|---|
| 0 | 셋업·데이터 검증 | ✅ | `step0_tag_analysis.ipynb` |
| 1 | tags 파싱·정규화 | ✅ | `step1_tag_analysis.ipynb`, `step1_tags_parsed.csv` |
| 2 | 카테고리별 Top 태그 | ✅ | `step2_tag_analysis.ipynb`, `step2_top_tags.csv` |
| 3 | 태그 효과·유의성 | ✅ | `step3_tag_analysis.ipynb`, `step3_tag_effect_top3.csv` |
| 4 | 시각화 | ⏳ 대기 | — |
| 5 | 모델 통합(선택) | ⏳ | — |
| 6 | 문서·발표 반영 | ⏳ | — |

---

## Step별 기록

### Step 0 — 셋업·검증
- 입력 `Track1/EDA/EDA_Phase2/cleaned_USvideos.csv` 로드, dedup 6249→6249(중복 없음), shape (6249, 26)
- **median Top3 = Gaming / Music / Film & Animation 확정 (match=True)** — Gaming 1.43M > Music 1.14M > Film & Animation 1.08M
- 타깃 n: **Gaming 101 / Music 792 / Film & Animation 306**
- 판단: Gaming 소표본이나 임계 적용 시 진행 가능 → 3개 유지

### Step 1 — tags 파싱·정규화
- 정규화: `|` 분리 · 따옴표/공백 제거 · 선두 `#` 제거 · 소문자 · `[none]`→빈 리스트
- 정합성: **`n_tags == tag_cnt` 비율 1.000** (6249 중 ~2개만 1 차이 — 무시 가능)
- 태그 없음 244개(3.9%), 평균 20.1개/영상
- `#` 제거로 해시태그/일반형 병합됨 (예: `#YouTubeRewind` = `youtuberewind`)
- 산출 `step1_tags_parsed.csv` (6249, 4): video_id·category·log_views·tags_joined

### Step 2 — 카테고리별 Top 태그
- document-frequency(영상당 set 중복 제거), 카테고리 내 ≥5 필터, 상위 15
- 임계 통과: **Gaming 32 / Music 338 / Film & Animation 130** → fallback 불필요
- **카테고리 vocabulary 거의 완전 분리**: 겹침 `Music ∩ Film & Animation = {official}` 단 1개, 3개 공통 0
- 노이즈/주의:
  - Music: 불용어 `the`(32), 일반 태그 `official`/`music`/`records`
  - Film & Animation: 단일 채널(ScreenJunkies) 시리즈 태그 편향 — `honest trailers`/`honest trailer`, `screenjunkies`/`screen junkies` 준중복
- 산출 `step2_top_tags.csv` (45, 3)

### Step 3 — 태그 효과·유의성
- in-category 보유 vs 미보유 log_views: diff + 양측 Mann-Whitney U + rank-biserial, 전체 BH-FDR(q)
- **FDR 생존: Gaming 0 / Music 4 / Film & Animation 3** (raw 1 / 6 / 8)
- 주요 효과 (diff = 자연로그 차이, exp = 배수):
  - **Film & Animation**: `disney` +1.46 (≈4.3배, rbc 0.50) · `movie` +0.99 · `trailer` +0.93 (FDR), `honest trailer(s)`·`screen junkies`·`comedy` (raw, 양수)
  - **Music**: `pop` +0.59 (≈1.8배) · `rca records label` +1.16 · `official video` +0.69 (양수) vs `alternative` −1.16 (≈0.31배) · `music` −0.90 · `country` −0.78 (FDR, 음수)
  - **Gaming**: FDR 0개. raw 유일 `games` −1.95 (n=8, 소표본 노이즈)
- 산출 `step3_tag_effect_top3.csv` (45, 13)

---

## 중간점검 — 유의미한가?

**판정: 부분적으로 유의미. Music·Film & Animation은 견고, Gaming은 사실상 빈손.**

핵심 해석:
1. **카테고리별 승자 태그 완전 분리** (Music=pop/메이저레이블, Film=disney/trailer, Gaming=없음) → **H4(효과의 카테고리별 이질성) 강력 지지**
2. **흔한 태그 ≠ 흥행**: Music `music`/`alternative`/`country`, Gaming 대부분 태그가 음수 → 히트작은 전형적 태그를 회피하는 **선택효과**. "이 태그 달면 뜬다"로 오독 금지
3. 최대 효과 `disney`(≈4.3배)는 **태그가 아니라 대형 IP/브랜드 효과** — 관찰적 연관(인과 아님), Part A "상관≠인과"와 같은 톤
4. **한계**: rank-biserial 대부분 |r|<0.3 (분포 겹침 큼), 유의성 일부는 큰 n 효과(pop n=166). **Gaming n=101 검정력 부족**

발표 프레이밍(권장):
- 헤드라인 = Music + Film & Animation 카테고리별 승자 태그
- Gaming = "소표본이라 태그 효과 검출 안 됨" → **한계 사례** 1줄 ('한계=발견' 톤)
- 메시지 = "흥행 태그는 카테고리마다 다르고, 흔한 태그가 곧 흥행은 아니다 (연관일 뿐)"

---

## 결정 사항 (Step 4 진입)

1. **`the` 제거**: 적용 — Step 2에 `STOPLIST = {'the'}` 추가 (Step 2·3 재실행 필요)
2. **Gaming**: 끝까지 추적 — 3개 카테고리 모두 시각화. Step 4 결과 보고 무의미하면 한계 사례로 유지
3. **채널 준중복**(`honest trailer(s)`/`screen junkies` 변형): 현재 유지, Step 4 시각화에서 재판단

---

**다음**: Step 2→3 재실행 후 Step 4(시각화) → `step4_fig_tag_effects.png`
