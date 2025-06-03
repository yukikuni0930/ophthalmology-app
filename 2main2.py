# 2main2.py

import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModel

# モデル設定
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"

# SentenceBERT風クラス定義
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded_input = self.tokenizer.batch_encode_plus(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask']).cpu()
            all_embeddings.extend(sentence_embeddings)
        return all_embeddings

# データ読み込み
df = pd.read_csv("question2.csv", header=0)
sentences = df["Questions"].fillna("").tolist()

# モデルでベクトル生成
model = SentenceBertJapanese(MODEL_NAME)
vectors = model.encode(sentences)

# JSONに保存
with open("question2.json", "w", encoding="utf-8") as f:
    json.dump([v.tolist() for v in vectors], f, ensure_ascii=False)

print("✅ ベクトルを question2.json に保存しました。")
