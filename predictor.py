"""
Inference wrapper for `best_model.pkl` (Run 1: fine-tuned MuRIL classifier).

Usage:
    from predictor import PredatoryClassifier
    clf = PredatoryClassifier("best_model.pkl")
    clf.predict(["akka what is your age", "semma dance performance"])

SECURITY: pickle can execute arbitrary code when loaded. Only load a .pkl you
created yourself (never one from an untrusted source).
"""
import os
import pickle
import re
import sys
import unicodedata

import numpy as np

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION_RE = re.compile(r"@\w+")


def clean_text(text):
    """
    Light normalisation mirroring the `clean_text` column used for training:
    lower-case, no URLs/@mentions/punctuation/emoji, single spaces.

    Letters, digits and *combining marks* are kept on purpose -- Tamil vowel signs
    are combining marks, so a naive `[^\\w\\s]` regex would silently destroy them.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = "".join(ch if unicodedata.category(ch)[0] in "LMNZ" else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_files(files, target_dir):
    """Write {relative_path: bytes} to disk once; reuse on later starts."""
    marker = os.path.join(target_dir, ".complete")
    if os.path.exists(marker):
        return target_dir
    os.makedirs(target_dir, exist_ok=True)
    root = os.path.abspath(target_dir)
    for rel, data in files.items():
        path = os.path.abspath(os.path.join(root, rel))
        if not path.startswith(root + os.sep):
            raise ValueError(f"Unsafe path in model bundle: {rel}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    open(marker, "w").close()
    return target_dir


class PredatoryClassifier:
    def __init__(self, pkl_path=None, cache_dir=None, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pkl_path = pkl_path or os.environ.get("MODEL_PATH", "best_model.pkl")
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        if payload.get("model_type") != "muril_finetuned_end_to_end":
            raise ValueError(
                f"This predictor is for Run 1 (muril_finetuned_end_to_end); "
                f"the pickle contains '{payload.get('model_type')}'."
            )

        self.run_name = payload.get("run_name", "MuRIL fine-tune")
        self.macro_f1 = payload.get("macro_f1")
        self.id2label = {int(k): v for k, v in payload["id2label"].items()}
        self.max_len = int(payload.get("max_len", 128))
        self.threshold = 0.5

        stamp = f"{os.path.getsize(pkl_path)}_{int(os.path.getmtime(pkl_path))}"
        cache_root = cache_dir or os.path.join(os.path.dirname(os.path.abspath(pkl_path)), ".model_cache", stamp)
        model_dir = _extract_files(payload["model_files"], os.path.join(cache_root, "model"))
        del payload  # free the ~1 GB of raw bytes

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

    def predict_proba(self, texts, batch_size=32, clean=True):
        """P(predatory) for each text."""
        if isinstance(texts, str):
            texts = [texts]
        prepared = [clean_text(t) if clean else str(t) for t in texts]
        out = []
        for i in range(0, len(prepared), batch_size):
            enc = self.tokenizer(prepared[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            out.append(self.torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def predict(self, texts, batch_size=32, clean=True):
        if isinstance(texts, str):
            texts = [texts]
        probs = self.predict_proba(texts, batch_size=batch_size, clean=clean)
        return [
            {
                "text": text,
                "label": self.id2label[int(prob >= self.threshold)],
                "predatory_probability": round(float(prob), 4),
            }
            for text, prob in zip(texts, probs)
        ]


if __name__ == "__main__":
    clf = PredatoryClassifier()
    for r in clf.predict(sys.argv[1:] or ["akka what is your age", "semma dance performance"]):
        print(r)
