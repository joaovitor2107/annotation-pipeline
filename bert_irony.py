"""
Treino e inferência do detector de ironia (irônico / não-irônico).

Fine-tune do BERTimbau para classificação binária de ironia.
Usa class weighting para lidar com o desbalanceamento severo (~7% irônicos).

Uso:
    # Treinar e avaliar:
    python bert_irony.py --step train
    python bert_irony.py --step eval

    # Ajustar freezing e peso da classe irônica:
    python bert_irony.py --step train --freeze-layers 4 --irony-weight-multiplier 3.0

    # Usar diretório diferente:
    python bert_irony.py --step train --model-dir models/meu_detector_ironia
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

IRONY_LABELS = ["not_ironic", "ironic"]
LABEL2ID = {l: i for i, l in enumerate(IRONY_LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}


@dataclass
class IronyConfig:
    model_name: str   = "neuralmind/bert-base-portuguese-cased"
    curated_csv: Path = Path("data/curated_tweets_stance.csv")
    model_dir: Path   = Path("models/bert_irony")
    results_dir: Path = Path("results")
    max_length: int   = 128
    epochs: int       = 5
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    infer_batch_size: int = 64
    weight_decay: float   = 0.01
    warmup_ratio: float   = 0.1
    early_stopping_patience: int = 2
    freeze_layers: int = 4
    irony_weight_multiplier: float = 1.0
    min_acceptable_f1: float = 0.50


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, max_length=max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def _device(infer: bool = False) -> torch.device:
    if infer:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class IronyDetector:
    def __init__(self, cfg: IronyConfig | None = None):
        self.cfg = cfg or IronyConfig()

    def _load_splits(self):
        df = pd.read_csv(self.cfg.curated_csv, dtype=str)
        df = df[df["irony"].isin(IRONY_LABELS)].copy()
        df["label_id"] = df["irony"].map(LABEL2ID)
        return (
            df[df["split"] == "train"],
            df[df["split"] == "val"],
            df[df["split"] == "test"],
        )

    def _freeze(self, model: BertForSequenceClassification) -> None:
        n = self.cfg.freeze_layers
        total = len(model.bert.encoder.layer)
        for param in model.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[total - n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.bert.pooler.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p   = sum(p.numel() for p in model.parameters())
        print(f"Freeze: top {n}/{total} layers + pooler + classifier "
              f"({trainable:,}/{total_p:,} params = {100*trainable/total_p:.1f}%)")

    def train(self):
        device = _device()
        print(f"Dispositivo: {device}")
        train_df, val_df, _ = self._load_splits()
        print(f"Treino: {len(train_df)}  Val: {len(val_df)}")
        print(f"Distribuição ironia (treino): {dict(train_df['irony'].value_counts())}")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, normalization=True)
        train_ds = TweetDataset(train_df["text"].tolist(), train_df["label_id"].tolist(),
                                tokenizer, self.cfg.max_length)
        val_ds   = TweetDataset(val_df["text"].tolist(),   val_df["label_id"].tolist(),
                                tokenizer, self.cfg.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.train_batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.infer_batch_size)

        counts  = train_df["label_id"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.sum() * len(IRONY_LABELS)
        weights[LABEL2ID["ironic"]] *= self.cfg.irony_weight_multiplier
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        print(f"Pesos de classe (not_ironic, ironic): {class_weights.tolist()}")

        model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(device)

        if self.cfg.freeze_layers > 0:
            self._freeze(model)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        total_steps  = self.cfg.epochs * len(train_loader)
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_f1, patience_count = 0.0, 0
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                loss = loss_fn(out.logits, batch["labels"].to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            val_f1, _ = self._eval(model, val_loader)
            print(f"Epoch {epoch}/{self.cfg.epochs} — loss={total_loss/len(train_loader):.4f}  val_F1={val_f1:.4f}")

            if val_f1 > best_f1 + 1e-4:
                best_f1, patience_count = val_f1, 0
                model.save_pretrained(self.cfg.model_dir)
                tokenizer.save_pretrained(self.cfg.model_dir)
                print(f"  → Novo melhor ({best_f1:.4f}), guardado.")
            else:
                patience_count += 1
                print(f"  → Sem melhoria ({patience_count}/{self.cfg.early_stopping_patience})")
                if patience_count >= self.cfg.early_stopping_patience:
                    print("  → Early stopping.")
                    break

        print(f"\nTreino concluído. Melhor val F1-macro: {best_f1:.4f}")
        return best_f1

    def evaluate(self):
        device = _device(infer=True)
        _, _, test_df = self._load_splits()
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, normalization=True)
        model = BertForSequenceClassification.from_pretrained(self.cfg.model_dir).to(device)
        test_ds = TweetDataset(test_df["text"].tolist(), test_df["label_id"].tolist(),
                               tokenizer, self.cfg.max_length)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.infer_batch_size)

        f1, report = self._eval(model, test_loader)
        print("\n" + report)
        print(f"F1-macro: {f1:.4f}")

        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.cfg.results_dir / "bert_irony_eval.json"
        with open(out, "w") as fp:
            json.dump({"f1_macro": f1}, fp, indent=2)
        print(f"Resultados em {out}")
        status = "PASS" if f1 >= self.cfg.min_acceptable_f1 else "WARN"
        print(f"{status}: F1-macro {f1:.3f}")
        return f1

    def _eval(self, model, loader):
        infer_device = _device(infer=True)
        train_device = next(model.parameters()).device
        model.to(infer_device).eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids=batch["input_ids"].to(infer_device),
                    attention_mask=batch["attention_mask"].to(infer_device),
                )
                preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
                labels.extend(batch["labels"].tolist())
        if str(train_device) != str(infer_device):
            model.to(train_device)
        f1     = f1_score(labels, preds, average="macro")
        report = classification_report(labels, preds, target_names=IRONY_LABELS, digits=3)
        return f1, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["train", "eval"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument("--irony-weight-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    cfg = IronyConfig(freeze_layers=args.freeze_layers,
                      irony_weight_multiplier=args.irony_weight_multiplier)
    if args.model_name:
        cfg.model_name = args.model_name
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)

    det = IronyDetector(cfg)
    steps = [args.step] if args.step else ["train", "eval"]
    for step in steps:
        print(f"\n{'='*60}\nPASSO: {step.upper()}\n{'='*60}")
        if step == "train":
            det.train()
        elif step == "eval":
            det.evaluate()
