from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import logging
from pathlib import Path
from typing import Union, List, Tuple

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, model_path="models/distilbert_finetuned"):
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

        with open(label_path, 'r') as f:
            label_map = json.load(f)

        self.labels = [None] * len(label_map)
        for intent, idx in label_map.items():
            self.labels[idx] = intent

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded. Device: {self.device}, Intents: {len(self.labels)}")

    def predict(self, text: str) -> Tuple[str, float]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)

        intent = self.labels[predicted_class.item()]
        confidence = confidence.item()

        return intent, confidence

    def is_ready(self) -> bool:
        try:
            self.predict("test")
            return True
        except Exception:
            return False