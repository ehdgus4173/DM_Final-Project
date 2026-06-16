"""Phase 2 - 2단계: VADER 감성 파생변수 추가.

features_partB.csv(ET 기준 확정 feature set)에 title 기반 VADER 감성 변수를 추가하여
features_partB_v2.csv로 저장한다. 누수 없음(title은 업로드 시점 확정).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

from nltk.sentiment import SentimentIntensityAnalyzer

OUT = Path(".")
feat = pd.read_csv(OUT / "features_partB.csv", parse_dates=["publish_time"])

# title은 features_partB.csv에 없으므로 Phase2 정제본에서 video_id로 가져온다
titles = pd.read_csv(Path("../EDA/EDA_Phase2/cleaned_USvideos.csv"),
                     usecols=["video_id", "title"]).drop_duplicates("video_id")
feat = feat.merge(titles, on="video_id", how="left")
print("title 결측:", feat["title"].isna().sum())

sia = SentimentIntensityAnalyzer()

def scores(s):
    d = sia.polarity_scores(str(s))
    return pd.Series([d["compound"], d["pos"], d["neg"]])

feat[["sent_compound", "sent_pos", "sent_neg"]] = feat["title"].apply(scores)
feat["sent_abs"] = feat["sent_compound"].abs()

sent_cols = ["sent_compound", "sent_pos", "sent_neg", "sent_abs"]
print("\n[감성 변수 요약]")
print(feat[sent_cols].describe().round(4).to_string())

# title은 모델에 직접 안 쓰므로 저장 시 제외 (감성 변수만 추가)
out = feat.drop(columns=["title"])
out.to_csv(OUT / "features_partB_v2.csv", index=False)
print("\nsaved -> features_partB_v2.csv", out.shape)
