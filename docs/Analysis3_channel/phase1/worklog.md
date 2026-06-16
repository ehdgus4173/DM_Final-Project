# Analysis 3 — Channel Track: 작업일지 (Worklog)

**담당**: Donghyun · **작성**: 2026-06-17, 발표 D-1
**전제**: `Background.md`(진단·근거) · `plan.md`(실행 계획)
**스크립트**: `Analysis3_channel/{build_channel_features, compare_channel, rf_channel, viz_channel}.py`

> Step 진행 중 결정·발견·보정 기록. 최종 종합은 `result.md`.

---

## Step 0 — channel_title 복구 & 머지 ✅

- **작업**: `dataset/USvideos.csv`(40,949행) → `video_id` dedup(`drop_duplicates(keep="last")`) → channel 매핑(6,351) → `cleaned_USvideos.csv`(6,249)에 left merge.
- **결과**:
  - **매칭률 100%** (6249/6249, 미매칭 0) → `__UNKNOWN__` fallback 미발동.
  - 고유 채널 **2,100개**, **단일 등장 61.6%**(1,294채널) → high-cardinality 확인(Background §5-4 예견).
  - 채널당 영상 수: median 1, mean 2.98, max 84(ESPN). Top: ESPN/TheEllenShow/Jimmy Fallon/Jimmy Kimmel/Netflix.
- **판단**: 복구 정합성 완벽. 단일 등장 다수 → OOF에서 카테고리 평균 fallback 필요(Step 1에서 확인).
- **산출**: `step0_channel_merge.csv` (6249×27).

## Step 1 — 채널 피처 (OOF target encoding) ✅

- **작업**: KFold(5, shuffle, rs=42)로 각 fold train에서만 채널 평균 산출 → held-out에 부여(자기 타깃 미포함). fallback: train fold 없는/단일 채널 → 카테고리 평균 → 글로벌.
- **결과**:
  - fallback **22.95%**(1,434건, 전부 카테고리평균; 글로벌 0건) — 단일 등장 채널이 OOF에서 자기 fold에 없을 때 발동.
  - **`chan_mean_oof` vs `log_views` 상관 = 0.551** (강력) / **`chan_mean_naive` = 0.865**.
  - OOF vs naive 상관 0.718 → OOF가 same-video 누수를 깎아낸 만큼 신호가 정직하게 낮아짐.
- **결정**: OOF·naive 둘 다 저장(Step 2 누수 시연용). `chan_freq`는 Step 3에서 `log1p` 변환.
- **산출**: `step1_channel_features.csv`.

## Step 2 — 누수 점검 (naive vs OOF) ✅

- naive 채널평균 투입 시 rand test R²=**0.737**, OOF는 **0.296**. 격차 **+0.44**가 곧 same-video 누수 낙관편향.
- → "OOF 안 하면 0.74로 부풀려진다"를 수치로 시연. **헤드라인은 OOF만** 사용(Background §5-3 준수).

## Step 3 — M_content vs M_content+channel ✅

- **M_content rand test R² = 0.1188** → A1 베이스라인 0.119 **정확히 재현**(features_partB_v2.csv로 재구성 충실성 확인, run_lr.ipynb 분실 영향 없음).
- **M_content+chan(OOF)**: rand 0.296 / CV 0.323±0.021 / **ΔR²(rand)=+0.177**, **ΔR²(CV)=+0.216**. std≪ΔR² → 견고.
- 채널 계수 전부 유의: `chan_mean_oof` +0.700, `chan_freq_log` +0.244 (p<.001).
- **tag_cnt 계수 반감**: 0.0218(M_content) → 0.0104(+chan) → "tag_cnt는 채널 프록시" 지지(Background §2-3).
- **time split R²**: −1.27 → −0.57 (개선되나 여전히 음수) → temporal drift는 채널로 안 풀림(별개 finding 유지).
- **잔여 분산 0.704** → A4 피벗 근거.
- **산출**: `step3_model_compare_channel.csv`.

## Step 4 — RF/GBM 병행 + permutation importance ✅

- RF: M_content test R²=0.112 → +chan **0.304**, ΔR²=+0.192 (LR과 일관 → 채널효과는 선형이 지배적, 비선형 추가이득 작음).
- **permutation importance: `chan_mean_oof` 1위(0.525, 압도) > `chan_freq_log` 2위 > `tag_cnt` 3위**.
- **tag_cnt 순위 강등: 1위(M_content) → 3위(+chan)** → A1 RF에서 최강이던 업로드시점 변수가 채널 뒤로 밀림. Background §2-3 "tag_cnt = 채널 운영수준 프록시" 직접 입증.
- **산출**: `step4_rf_importance.csv`, `step4_fig_importance.png`.

## Step 5 — 시각화 ✅

- (좌) CV R² 점프 0.107→0.323(±std 에러바, ΔR²=+0.216), (우) 설명 30% vs 잔여 70%(A4 피벗).
- 한글 폰트(Malgun Gothic) 지정으로 글리프 깨짐 해결.
- **산출**: `step5_fig_channel_r2.png`.

---

## 보정·이슈 메모

- **Analysis3_channel/.git 0바이트 잔여물**: 착수 시점 자동 소멸(일시 아티팩트). 별도 조치 불요.
- **R² 점프가 Background 예상(0.4~0.6)보다 작음**: OOF 헤드라인 0.30. naive(누수)면 0.74로 예상 범위 안. Background §5-1대로 **아크 불변** — "최강 업로드시점 신호도 천장" → A4 근거로 동일 사용. 정직성상 OOF 보고가 맞음.
- **fallback 23%**: R² 기여는 반복 등장 대형 채널 꼬리에서 주로 발생, 단일 등장 채널은 카테고리 평균으로 회귀(약신호) — result 한계에 명시.

---

**다음 문서**: `result.md` — 최종 종합·해석·발표 반영
