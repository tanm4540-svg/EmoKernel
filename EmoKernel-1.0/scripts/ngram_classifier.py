import numpy as np
import hashlib
import pickle
import os

class NGramClassifier:
    def __init__(self, dim=8192, ngram_range=(1, 4)):
        self.dim = dim
        self.ngram_range = ngram_range
        self.category_vectors = {}
        self.category_names = []

    def _hash_feat(self, ngram):
        h = hashlib.md5(ngram.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def _text_to_vec(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                idx = self._hash_feat(text[i:i+n])
                vec[idx] += 1.0
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 1e-10 else vec

    def predict(self, text, top_k=5, min_sim=0.0):
        vec = self._text_to_vec(text)
        scores = {}
        for cat, cv in self.category_vectors.items():
            sim = float(np.dot(vec, cv))
            if sim > min_sim:
                scores[cat] = round(sim, 4)
        items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        if items:
            total = sum(s[1] for s in items)
            return {k: round(v/total, 4) for k, v in items}
        return {}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(dim=data["dim"], ngram_range=data["ngram_range"])
        obj.category_vectors = data["category_vectors"]
        obj.category_names = data["category_names"]
        return obj


def get_default_classifier():
    path = os.path.join(os.path.dirname(__file__), "ngram_model.pkl")
    if os.path.exists(path):
        return NGramClassifier.load(path)
    return None
