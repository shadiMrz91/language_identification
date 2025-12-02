import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

class EmbeddingExtractor:

    MODEL_CHECKPOINTS = {
        "labse": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "distilmbert": "distilbert-base-multilingual-cased"
    }

    def __init__(self, model_name="labse", device=None):
        self.model_name = model_name.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model_ckpt = self.MODEL_CHECKPOINTS[self.model_name]
        except KeyError:
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. "
                f"Available options: {list(self.MODEL_CHECKPOINTS.keys())}"
            )

        self._load_model()



    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt).to(self.device)

    def compute_embeddings(self, texts, batch_size=32, max_length=512):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {self.model_name}"):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output.cpu().numpy()
                else:
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(emb)
        return np.vstack(embeddings)

    def _save_embeddings(self, embeddings, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)
        print(f" Embeddings saved: {path}")

    def _load_embeddings(self, path):
        print(f" Loading existing embeddings: {path}")
        return np.load(path)

    def _get_embedding_path(self, split_name):
        return f"data/embeddings/{self.model_name}/{split_name}_{self.model_name}.npy"

    def load_or_compute_embeddings(self, texts, split_name, overwrite=False):
        path = self._get_embedding_path(split_name)
        if os.path.exists(path) and not overwrite:
            return self._load_embeddings(path)

        print(f"Computing embeddings for {split_name} split...")
        emb = self.compute_embeddings(texts)
        self._save_embeddings(emb, path)
        return emb

