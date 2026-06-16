"""R² 천장 진단: 어떤 변수군을 넣으면 R²가 얼마나 오르는지 실측.
- 누수 경계를 명확히 구분하여 비교 (업로드시점 / 트렌딩메타 / 사후참여도)
- LR(OLS)과 RF·GBM을 같은 feature set으로 비교
- 성능은 랜덤 split(베이스라인) 기준
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv(Path("../../dataset/cleaned_USvideos.csv"))
df = df.sort_values("views").drop_duplicates("video_id", keep="last").reset_index(drop=True)

title_obj = ["title_len", "caps_ratio", "exclaim_cnt", "question_cnt",
             "has_number", "has_bracket", "tag_cnt"]
time_cat = ["publish_hour", "publish_weekday"]
trending_meta = ["trending_lag", "days_on_trending"]
engagement = ["log_likes", "log_comment_count", "dislikes",
              "comments_disabled", "ratings_disabled"]

# bool -> int
for c in ["comments_disabled", "ratings_disabled"]:
    df[c] = df[c].astype(int)

FEATURE_SETS = {
    "S0  업로드시점만 (현재 H4 모델)": title_obj + time_cat,
    "S1  +트렌딩 메타(lag,days)":      title_obj + time_cat + trending_meta,
    "S2  +사후 참여도(likes 등)=누수":  title_obj + time_cat + trending_meta + engagement,
}

y = df["log_views"].values
tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)


def eval_lr(feats):
    # category 원핫 포함 OLS
    f = "log_views ~ " + " + ".join(feats) + " + C(category)"
    m = smf.ols(f, data=df.iloc[tr_idx]).fit()
    te = df.iloc[te_idx]
    return r2_score(te["log_views"], m.predict(te))


def eval_tree(feats, model):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["category"])],
        remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("m", model)])
    X = df[feats + ["category"]]
    pipe.fit(X.iloc[tr_idx], y[tr_idx])
    return r2_score(y[te_idx], pipe.predict(X.iloc[te_idx]))


print(f"{'feature set':36s}  {'LR':>8} {'RF':>8} {'GBM':>8}")
print("-" * 64)
for name, feats in FEATURE_SETS.items():
    lr = eval_lr(feats)
    rf = eval_tree(feats, RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    gb = eval_tree(feats, GradientBoostingRegressor(random_state=42))
    print(f"{name:36s}  {lr:8.3f} {rf:8.3f} {gb:8.3f}")
