"""Phase 2 - 2단계: M0(객관만) vs M1(+VADER 감성) 비교 + H4 재확인."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

OUT = Path(".")
df = pd.read_csv(OUT / "features_partB_v2.csv", parse_dates=["publish_time"])
hour_labels = ["00-05", "06-11", "12-17", "18-23"]
df["hour_bin"] = pd.Categorical(df["hour_bin"], categories=hour_labels)

obj_feats = ["title_len", "caps_ratio", "exclaim_cnt", "question_cnt",
             "has_number", "has_bracket", "tag_cnt"]
sent_feats = ["sent_compound", "sent_pos", "sent_neg"]  # sent_abs는 compound와 중복 → 제외

base = (" + ".join(obj_feats) +
        " + C(hour_bin, Treatment(reference='18-23')) + C(publish_weekday) + C(category)")
F_M0 = "log_views ~ " + base
F_M1 = "log_views ~ " + " + ".join(sent_feats) + " + " + base


def time_split(d, frac=0.8):
    d = d.dropna(subset=["publish_time"]).sort_values("publish_time").reset_index(drop=True)
    cut = int(len(d) * frac)
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def perf(formula, tr, te):
    m = smf.ols(formula, data=tr).fit()
    te = te[te["category"].isin(set(tr["category"].unique()))]
    p = m.predict(te)
    return (m.rsquared,
            r2_score(te["log_views"], p),
            np.sqrt(mean_squared_error(te["log_views"], p)))


tr_t, te_t = time_split(df)
tr_r, te_r = train_test_split(df.dropna(subset=["publish_time"]), test_size=0.2, random_state=42)

print("=" * 60)
print("M0 vs M1 성능 비교  (R2 train / R2 test / RMSE test)")
print("=" * 60)
rows = []
for name, F in [("M0 (객관만)", F_M0), ("M1 (+감성)", F_M1)]:
    r2tr_r, r2te_r, rmse_r = perf(F, tr_r, te_r)   # 랜덤 split (베이스라인)
    r2tr_t, r2te_t, rmse_t = perf(F, tr_t, te_t)   # 시간 split (drift)
    rows.append({"model": name,
                 "rand_R2_train": round(r2tr_r, 4), "rand_R2_test": round(r2te_r, 4),
                 "rand_RMSE_test": round(rmse_r, 4),
                 "time_R2_test": round(r2te_t, 4), "time_RMSE_test": round(rmse_t, 4)})
cmp = pd.DataFrame(rows)
print(cmp.to_string(index=False))
dR2 = cmp.loc[1, "rand_R2_test"] - cmp.loc[0, "rand_R2_test"]
print(f"\nΔR² (랜덤 split test, M1 - M0) = {dR2:+.4f}")
cmp.to_csv(OUT / "model_compare_sentiment.csv", index=False)

print("\n" + "=" * 60)
print("M1 감성변수 계수 / p-value (train, 시간 split)")
print("=" * 60)
m1 = smf.ols(F_M1, data=tr_t).fit()
tbl = pd.DataFrame({"coef": m1.params, "p_value": m1.pvalues}).loc[sent_feats]
tbl["sig"] = np.where(tbl["p_value"] < 0.05, "*", "")
print(tbl.round(4).to_string())

print("\n" + "=" * 60)
print("VADER vs 객관 지표 상관 (이미 커버하는지 점검)")
print("=" * 60)
print(df[["sent_compound", "sent_abs", "caps_ratio", "exclaim_cnt"]].corr().round(3).to_string())

print("\n" + "=" * 60)
print("H4 — Top3 카테고리별 감성변수 계수 (* p<0.05)")
print("=" * 60)
top3 = df["category"].value_counts().head(3).index.tolist()
sub = ("log_views ~ " + " + ".join(sent_feats + obj_feats) +
       " + C(hour_bin, Treatment(reference='18-23')) + C(publish_weekday)")
def star(p): return "*" if p < 0.05 else ""
models = {c: smf.ols(sub, data=df[df["category"] == c]).fit() for c in top3}
out = []
for f in sent_feats:
    r = {"feature": f}
    for c in top3:
        m = models[c]
        r[c] = f"{m.params[f]:.3f}{star(m.pvalues[f])}" if f in m.params.index else "-"
    out.append(r)
print("Top3:", top3)
print(pd.DataFrame(out).set_index("feature").to_string())
