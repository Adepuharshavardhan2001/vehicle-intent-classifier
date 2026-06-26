import pandas as pd
import numpy as np
import json
import os
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "data_path": "data/dataset.xlsx",
    "output_dir": "models/distilbert_finetuned",
    "label_map_path": "models/label_map.json",
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 5e-5,
    "test_size": 0.2,
    "random_seed": 42,
    "max_length": 128,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(CONFIG["random_seed"])
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG["data_path"]):
        raise FileNotFoundError(f"Dataset not found: {CONFIG['data_path']}")

    print("Loading dataset...")
    df = pd.read_excel(CONFIG["data_path"], sheet_name='Dataset')
    df = df[['Utterance', 'Intent']].dropna()

    label_map = {label: idx for idx, label in enumerate(df['Intent'].unique())}
    df['label'] = df['Intent'].map(label_map)

    with open(CONFIG["label_map_path"], 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"Loaded {len(df)} samples, {len(label_map)} intents")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Utterance'].tolist(), df['label'].tolist(),
        test_size=CONFIG["test_size"], random_state=CONFIG["random_seed"]
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=CONFIG["max_length"])
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=CONFIG["max_length"])

    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_encodings['input_ids']),
        torch.tensor(val_encodings['attention_mask']),
        torch.tensor(val_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=len(label_map)
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels_list, all_preds)
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc*100:.2f}%")

    print(f"\n{classification_report(all_labels_list, all_preds, zero_division=0)}")
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"Model saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()