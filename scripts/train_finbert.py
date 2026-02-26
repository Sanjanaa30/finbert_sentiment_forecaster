import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "ProsusAI/finbert"
OUT_DIR = "models/finbert_sentiment"
PHRASEBANK_CSV = Path("data/processed/phrasebank_allagree.csv")
ARTIFACTS_DIR = Path("artifacts")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    if not PHRASEBANK_CSV.exists():
        raise FileNotFoundError(
            f"Missing training data: {PHRASEBANK_CSV}. Run scripts/load_phrasebank.py first."
        )

    df = pd.read_csv(PHRASEBANK_CSV)

    # Keep only expected columns and map labels to class ids.
    df = df[["sentence", "label"]].dropna()
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label"].map(label_map)
    if df["label"].isna().any():
        bad_labels = sorted(df[df["label"].isna()]["label"].dropna().unique())
        raise ValueError(f"Unexpected label values in phrasebank data: {bad_labels}")
    df["label"] = df["label"].astype(int)

    ds = Dataset.from_pandas(df, preserve_index=False)

    # split train/val
    ds = ds.train_test_split(test_size=0.15, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.select_columns(cols)
    val_ds = val_ds.select_columns(cols)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=False,  # set True if you have a compatible GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,

        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    # Save validation logits + labels for temperature scaling step
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = trainer.predict(val_ds)
    np.save(ARTIFACTS_DIR / "val_logits.npy", preds.predictions)
    np.save(ARTIFACTS_DIR / "val_labels.npy", preds.label_ids)

    print("Saved fine-tuned model ->", OUT_DIR)
    print("Saved val_logits/val_labels -> artifacts/")


if __name__ == "__main__":
    main()
