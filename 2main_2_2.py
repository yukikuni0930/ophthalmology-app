# main_final.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import json
import re

# データ読み込み
df = pd.read_csv("question2.csv", header=0)
with open("question2.json", "r", encoding="utf-8") as f:
    loaded_vecs = np.array(json.load(f))  # 500行のベクトル

# タイトル
st.title("眼科専門医試験 類似問題検索アプリ")

# 選択肢の生成
unique_number_of_examinations = df["number of examinations"].dropna().unique()
unique_subtitles = df["Subtitle"].dropna().unique()

# ユーザー選択
selected_number = st.selectbox("回数", unique_number_of_examinations)
selected_subtitle = st.selectbox("問題番号", unique_subtitles)

# フィルタリング
filtered_df = df[
    (df["number of examinations"] == selected_number) &
    (df["Subtitle"] == selected_subtitle)
]

if filtered_df.empty:
    st.error("該当する問題が見つかりません。")
    st.stop()

# 対象のインデックスとベクトル取得
target_index = filtered_df.index[0]
target_vec = torch.tensor(loaded_vecs[target_index]).unsqueeze(0)

# 全問題との類似度計算
all_vecs = torch.tensor(loaded_vecs)
similarity_scores = F.cosine_similarity(target_vec, all_vecs).tolist()

# 結果をDataFrameに追加
df["類似度"] = similarity_scores
df_results = df.sort_values(by="類似度", ascending=False)

# 類似問題上位10件を表示
st.subheader("類似度の高い問題トップ10")
st.dataframe(df_results[["類似度", "number of examinations", "Subtitle", "Questions"]].head(10))
