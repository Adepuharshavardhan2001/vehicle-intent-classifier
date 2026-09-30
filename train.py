import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "data_path": "data/dataset.xlsx",
    "output_dir": "models/distilbert_finetuned",
    "label_map_path": "models/label_map.json",
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "random_seed": 42,
    "max_length": 128,
    "early_stopping_patience": 2,
    "warmup_ratio": 0.1,
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
    df = pd.read_excel(CONFIG["data_path"], sheet_name="Dataset")
    df = df[["Utterance", "Intent"]].dropna()

    unique_labels = sorted(df["Intent"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df["label"] = df["Intent"].map(label_map)

    with open(CONFIG["label_map_path"], "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Loaded {len(df)} samples, {len(label_map)} intents")
    print(f"Class distribution:\n{df['Intent'].value_counts()}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["Utterance"].tolist(),
        df["label"].tolist(),
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"],
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=CONFIG["max_length"],
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=CONFIG["max_length"],
    )

    train_dataset = TensorDataset(
        torch.tensor(train_encodings["input_ids"]),
        torch.tensor(train_encodings["attention_mask"]),
        torch.tensor(train_labels),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_encodings["input_ids"]),
        torch.tensor(val_encodings["attention_mask"]),
        torch.tensor(val_labels),
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=len(label_map)
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    num_training_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(CONFIG["warmup_ratio"] * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

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
            scheduler.step()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(all_labels_list, all_preds)
        print(
            f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f} "
            f"- Val Loss: {avg_val_loss:.4f} - Val Acc: {acc*100:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = acc
            model.save_pretrained(CONFIG["output_dir"])
            tokenizer.save_pretrained(CONFIG["output_dir"])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    metadata = {
        "model_name": CONFIG["model_name"],
        "num_labels": len(label_map),
        "labels": list(label_map.keys()),
        "val_accuracy": float(best_val_acc),
        "val_loss": float(best_val_loss),
        "epochs_trained": epoch + 1,
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "max_length": CONFIG["max_length"],
        "trained_at": datetime.utcnow().isoformat(),
    }
    with open(f"{CONFIG['output_dir']}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{classification_report(all_labels_list, all_preds, zero_division=0)}")
    print(f"Model saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()