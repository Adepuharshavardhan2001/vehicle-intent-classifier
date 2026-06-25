from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

class Classifier:
    def __init__(self, model_path="models/distilbert_finetuned"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load the intent mapping saved from training
        with open('models/label_map.json', 'r') as f:
            label_map = json.load(f)
        
        # Convert {"navigate_to_poi": 0} to ["navigate_to_poi", ...]
        self.labels = [None] * len(label_map)
        for intent, idx in label_map.items():
            self.labels[idx] = intent

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
        
        return self.labels[predicted_class.item()], confidence.item()