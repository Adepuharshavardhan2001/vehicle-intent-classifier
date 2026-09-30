import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))


class Classifier:
    def __init__(self, model_path: str = "models/distilbert_finetuned"):
        self.base_dir = Path(__file__).parent.parent
        model_path = os.getenv("MODEL_PATH", str(self.base_dir / model_path))
        label_path = os.getenv("LABEL_PATH", str(self.base_dir / "models/label_map.json"))

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label map not found: {label_path}")

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        with open(label_path, "r") as f:
            label_map = json.load(f)

        self.labels = [None] * len(label_map)
        for intent, idx in label_map.items():
            self.labels[idx] = intent

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self._ready_cache: bool | None = None
        logger.info(
            f"Model loaded. Device: {self.device}, "
            f"Intents: {len(self.labels)}, Max length: {MAX_LENGTH}"
        )

    def predict(self, text: str) -> Tuple[str, float]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)

        intent = self.labels[predicted_class.item()]
        confidence = confidence.item()

        return intent, confidence

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        if not texts or any(not t or not t.strip() for t in texts):
            raise ValueError("Input texts cannot be empty")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidences, predicted_classes = torch.max(probs, dim=-1)

        return [
            (self.labels[pc.item()], c.item())
            for pc, c in zip(predicted_classes, confidences)
        ]

    def is_ready(self) -> bool:
        if self._ready_cache is None:
            try:
                self.predict("test")
                self._ready_cache = True
            except Exception:
                self._ready_cache = False
        return self._ready_cache