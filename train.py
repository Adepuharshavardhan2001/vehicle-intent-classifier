import pandas as pd
import numpy as np
import json
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 1. CONFIGURATION
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/dataset.xlsx"  # <--- CHANGED TO MATCH YOUR FILENAME
OUTPUT_DIR = "models/distilbert_finetuned"
LABEL_MAP_PATH = "models/label_map.json"

# 2. LOAD YOUR 1,500 ROW DATASET
print(" Loading In-Car Intent Dataset...")
df = pd.read_excel(DATA_PATH, sheet_name='Dataset') 
df = df[['Utterance', 'Intent']].dropna()

# 3. PREPARE LABELS
label_map = {label: idx for idx, label in enumerate(df['Intent'].unique())}
df['label'] = df['Intent'].map(label_map)

os.makedirs("models", exist_ok=True)
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)

print(f" Loaded {len(df)} samples.")
print(f" Unique Intents: {len(label_map)}")

# 4. SPLIT DATA (80% Train, 20% Validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Utterance'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 5. TOKENIZE
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(" Tokenizing text data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# 6. CONVERT TO PYTORCH TENSORS
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels_tensor = torch.tensor(train_labels)

val_input_ids = torch.tensor(val_encodings['input_ids'])
val_attention_mask = torch.tensor(val_encodings['attention_mask'])
val_labels_tensor = torch.tensor(val_labels)

# 7. CREATE DATALOADERS
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels_tensor)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 8. INITIALIZE MODEL & OPTIMIZER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

print("\n Starting Training...\n")

# 9. TRAINING LOOP
model.train()
for epoch in range(5): 
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
    
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 10. EVALUATE EACH EPOCH
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/5 - Avg Loss: {total_loss/len(train_loader):.4f} - Val Accuracy: {acc*100:.2f}%")
    model.train()

# 11. FINAL EVALUATION & SAVE
print("\n Training Complete. Final Evaluation on Validation Set:")


print(classification_report(all_labels, all_preds, zero_division=0))

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n Model saved to: {OUTPUT_DIR}")
print(" Training complete! You can now run Docker.")