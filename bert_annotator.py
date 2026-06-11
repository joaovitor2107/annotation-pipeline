"""
Fine-tune do BERTimbau para classificação de stance e anotação em larga escala.

Classifica tweets em três categorias:
  - a favor  : apoia a posição/evento em análise
  - contra   : critica ou condena
  - neutro   : relata factos ou não toma partido

Inclui três passos:
  train    — fine-tune com os tweets curados
  eval     — avalia no conjunto de teste
  annotate — classifica o corpus completo (com filtro de ironia opcional)

Uso:
    python bert_annotator.py --step train
    python bert_annotator.py --step eval
    python bert_annotator.py --step annotate \
        --tweets-dir data/tweets/ \
        --annotation-output data/pseudo_labeled.csv

    # Com freeze-layers e filtro de ironia:
    python bert_annotator.py --step train --freeze-layers 4
    python bert_annotator.py --step annotate \
        --model-dir models/bert_stance_freeze4 \
        --irony-filter \
        --tweets-dir data/tweets/ \
        --annotation-output data/pseudo_labeled_noirony.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
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
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data_selection import STANCE_LABELS
from tweet_stream import RawTweetStream

LABEL2ID = {lbl: i for i, lbl in enumerate(STANCE_LABELS)}
ID2LABEL  = {i: lbl for lbl, i in LABEL2ID.items()}


@dataclass
class BERTConfig:
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    curated_csv: Path = Path("data/curated_tweets_stance.csv")
    model_dir: Path   = Path("models/bert_stance")
    results_dir: Path = Path("results")
    annotation_output: Path = Path("data/pseudo_labeled_stance.csv")
    # Dados raw para anotação
    raw_data_dirs: list[Path] = field(default_factory=list)
    # Treino
    max_length: int   = 128
    epochs: int       = 5
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    weight_decay: float  = 0.01
    early_stopping_patience: int = 2
    warmup_ratio: float  = 0.1
    scheduler_type: str  = "linear"
    freeze_layers: int   = 0
    # Inferência
    infer_batch_size: int = 64
    max_tweets: int = 2_350_000
    min_acceptable_f1: float = 0.60
    # Filtro de ironia
    irony_filter: bool = False
    irony_model_dir: Path  = Path("models/bert_irony")
    irony_threshold: float = 0.10


class StanceDataset(Dataset):
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


def _get_device(infer: bool = False) -> torch.device:
    # BertForSequenceClassification tem um bug numérico em inferência no MPS
    # (Apple Silicon): argmax produz resultados errados. Usar sempre CPU.
    if infer:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BERTStanceAnnotator:
    def __init__(self, cfg: BERTConfig | None = None):
        self.cfg = cfg or BERTConfig()

    def _apply_layer_freezing(self, model: BertForSequenceClassification) -> None:
        n = self.cfg.freeze_layers
        total_layers = len(model.bert.encoder.layer)
        freeze_up_to = total_layers - n
        for param in model.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[freeze_up_to:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.bert.pooler.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Freeze: top {n}/{total_layers} layers + pooler + classifier "
              f"({trainable:,}/{total:,} params = {100*trainable/total:.1f}%)")

    # ------------------------------------------------------------------
    # Passo 1: Fine-tune
    # ------------------------------------------------------------------

    def train(self):
        device = _get_device()
        print(f"Dispositivo de treino: {device}")

        train_df, val_df, _ = self._load_splits()
        print(f"Treino: {len(train_df)}  Val: {len(val_df)}")
        print(f"Distribuição treino: {dict(train_df['label'].value_counts())}")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, normalization=True)
        train_ds = StanceDataset(train_df["text"].tolist(), train_df["label_id"].tolist(),
                                 tokenizer, self.cfg.max_length)
        val_ds   = StanceDataset(val_df["text"].tolist(),   val_df["label_id"].tolist(),
                                 tokenizer, self.cfg.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.train_batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.infer_batch_size)

        counts  = train_df["label_id"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.sum() * len(STANCE_LABELS)
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        print(f"Pesos de classe: {class_weights.tolist()}")

        model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=len(STANCE_LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(device)

        if self.cfg.freeze_layers > 0:
            self._apply_layer_freezing(model)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        total_steps  = self.cfg.epochs * len(train_loader)
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        if self.cfg.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_f1, patience_count = 0.0, 0
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                loss = loss_fn(outputs.logits, batch["labels"].to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            infer_device = _get_device(infer=True)
            val_f1, _ = self._eval_loader(model, val_loader, device, infer_device)
            print(f"Epoch {epoch}/{self.cfg.epochs} — loss={total_loss/len(train_loader):.4f}  val_F1={val_f1:.4f}")

            if val_f1 > best_f1 + 1e-4:
                best_f1, patience_count = val_f1, 0
                model.save_pretrained(self.cfg.model_dir)
                tokenizer.save_pretrained(self.cfg.model_dir)
                print(f"  → Novo melhor ({best_f1:.4f}), salvo em {self.cfg.model_dir}")
            else:
                patience_count += 1
                print(f"  → Sem melhoria ({patience_count}/{self.cfg.early_stopping_patience})")
                if patience_count >= self.cfg.early_stopping_patience:
                    print("  → Early stopping.")
                    break

        print(f"\nTreino concluído. Melhor val F1-macro: {best_f1:.4f}")
        return best_f1

    # ------------------------------------------------------------------
    # Passo 2: Avaliação no conjunto de teste
    # ------------------------------------------------------------------

    def evaluate(self):
        device = _get_device(infer=True)
        if not (self.cfg.model_dir / "config.json").exists():
            print(f"ERRO: modelo não encontrado em {self.cfg.model_dir}.")
            return

        _, _, test_df = self._load_splits()
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, normalization=True)
        model = BertForSequenceClassification.from_pretrained(self.cfg.model_dir).to(device)
        test_ds = StanceDataset(test_df["text"].tolist(), test_df["label_id"].tolist(),
                                tokenizer, self.cfg.max_length)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.infer_batch_size)

        f1, report = self._eval_loader(model, test_loader)
        print(f"\nTest set: {len(test_df)} exemplos")
        print("\n" + report)
        print(f"F1-macro: {f1:.4f}")

        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.cfg.results_dir / "bert_stance_eval.json"
        with open(out, "w") as fp:
            json.dump({"f1_macro": f1}, fp, indent=2)
        print(f"Resultados em {out}")
        status = "PASS" if f1 >= self.cfg.min_acceptable_f1 else "AVISO"
        print(f"{status}: F1-macro {f1:.3f}")
        return {"f1_macro": f1, "report": report}

    # ------------------------------------------------------------------
    # Passo 3: Anotação do corpus completo
    # ------------------------------------------------------------------

    def annotate(self):
        if not self.cfg.raw_data_dirs:
            print("ERRO: --tweets-dir é obrigatório para o passo annotate.")
            return
        if not (self.cfg.model_dir / "config.json").exists():
            print(f"ERRO: modelo não encontrado em {self.cfg.model_dir}.")
            return

        device = _get_device(infer=True)
        print(f"Carregando modelo de stance de {self.cfg.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, normalization=True)
        model = BertForSequenceClassification.from_pretrained(self.cfg.model_dir).to(device)
        model.eval()

        irony_model, irony_tokenizer = None, None
        if self.cfg.irony_filter:
            if not (self.cfg.irony_model_dir / "config.json").exists():
                print(f"ERRO: modelo de ironia não encontrado em {self.cfg.irony_model_dir}.")
                return
            print(f"Carregando modelo de ironia de {self.cfg.irony_model_dir} "
                  f"(threshold={self.cfg.irony_threshold})...")
            irony_tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, normalization=True)
            irony_model = BertForSequenceClassification.from_pretrained(
                self.cfg.irony_model_dir).to(device)
            irony_model.eval()

        skip_ids: set[str] = set()
        if self.cfg.curated_csv.exists():
            df_cur = pd.read_csv(self.cfg.curated_csv, usecols=["conversation_id"], dtype=str)
            skip_ids.update(df_cur["conversation_id"].str.strip().dropna())
            print(f"Ignorando {len(skip_ids)} IDs curados.")

        out_path = self.cfg.annotation_output
        if out_path.exists():
            df_prev = pd.read_csv(out_path, usecols=["conversation_id"], dtype=str)
            prev_ids = set(df_prev["conversation_id"].str.strip().dropna())
            skip_ids.update(prev_ids)
            print(f"Retomando: {len(prev_ids)} já anotados, {len(skip_ids)} total a pular.")
            write_header = False
        else:
            write_header = True

        stream = RawTweetStream(self.cfg.raw_data_dirs, skip_ids,
                                chunk_size=10_000, max_tweets=self.cfg.max_tweets)

        total_written = 0
        total_irony_skipped = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "a", newline="") as out_file:
            if write_header:
                out_file.write("conversation_id,text,predicted_label,confidence\n")

            batch_texts: list[str] = []
            batch_ids:   list[str] = []

            def flush_batch():
                nonlocal total_written, total_irony_skipped
                if not batch_texts:
                    return

                active_texts = batch_texts[:]
                active_ids   = batch_ids[:]

                if irony_model is not None:
                    irony_enc = irony_tokenizer(
                        active_texts, max_length=self.cfg.max_length,
                        padding="max_length", truncation=True, return_tensors="pt",
                    )
                    with torch.no_grad():
                        irony_logits = irony_model(
                            input_ids=irony_enc["input_ids"].to(device),
                            attention_mask=irony_enc["attention_mask"].to(device),
                        ).logits.cpu()
                    irony_probs = torch.softmax(irony_logits, dim=-1).numpy()
                    irony_prob_ironic = irony_probs[:, 1]  # índice 1 = "ironic"
                    keep_mask = irony_prob_ironic < self.cfg.irony_threshold
                    total_irony_skipped += int((~keep_mask).sum())
                    active_texts = [t for t, k in zip(active_texts, keep_mask) if k]
                    active_ids   = [i for i, k in zip(active_ids,   keep_mask) if k]

                if not active_texts:
                    batch_texts.clear(); batch_ids.clear()
                    return

                enc = tokenizer(
                    active_texts, max_length=self.cfg.max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                with torch.no_grad():
                    logits = model(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                    ).logits.cpu()
                probs    = torch.softmax(logits, dim=-1).numpy()
                pred_ids = probs.argmax(axis=1)

                for cid, text, pid, prob_row in zip(active_ids, active_texts, pred_ids, probs):
                    label      = ID2LABEL[pid]
                    confidence = float(prob_row[pid])
                    escaped    = text.replace('"', '""')
                    out_file.write(f'{cid},"{escaped}",{label},{confidence:.4f}\n')

                total_written += len(active_texts)
                batch_texts.clear(); batch_ids.clear()

            for chunk_df in stream:
                for _, row in chunk_df.iterrows():
                    cid  = str(row.get("conversation_id", "")).strip()
                    text = str(row.get("text", "")).strip()
                    if not cid or not text:
                        continue
                    batch_texts.append(text)
                    batch_ids.append(cid)
                    if len(batch_texts) >= self.cfg.infer_batch_size:
                        flush_batch()
                        processed = total_written + total_irony_skipped
                        if processed % 10_000 < self.cfg.infer_batch_size:
                            print(f"  Anotados {total_written:,} | "
                                  f"removidos por ironia {total_irony_skipped:,}...")
            flush_batch()

        print(f"\nConcluído. Total anotado: {total_written:,}")
        if self.cfg.irony_filter:
            total_seen = total_written + total_irony_skipped
            pct = 100 * total_irony_skipped / total_seen if total_seen else 0
            print(f"Filtrado por ironia: {total_irony_skipped:,} tweets ({pct:.1f}%)")
        print(f"Saída: {out_path}")

        if total_written > 0:
            df_out = pd.read_csv(out_path, dtype=str)
            dist   = df_out["predicted_label"].value_counts()
            print("\nDistribuição de labels:")
            for lbl, cnt in dist.items():
                print(f"  {lbl}: {cnt:,} ({100*cnt/len(df_out):.1f}%)")

    # ------------------------------------------------------------------

    def _load_splits(self):
        df = pd.read_csv(self.cfg.curated_csv, dtype=str)
        df = df[df["label"].isin(STANCE_LABELS)].copy()
        df["label_id"] = df["label"].map(LABEL2ID)
        return (
            df[df["split"] == "train"],
            df[df["split"] == "val"],
            df[df["split"] == "test"],
        )

    def _eval_loader(self, model, loader, train_device=None, infer_device=None):
        if infer_device is None:
            infer_device = _get_device(infer=True)
        model_was_on = next(model.parameters()).device
        model.to(infer_device).eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(infer_device),
                    attention_mask=batch["attention_mask"].to(infer_device),
                )
                all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch["labels"].tolist())
        if train_device is not None and str(model_was_on) != str(infer_device):
            model.to(train_device)
        f1     = f1_score(all_labels, all_preds, average="macro")
        report = classification_report(all_labels, all_preds,
                                        target_names=STANCE_LABELS, digits=3)
        return f1, report


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTimbau para classificação de stance em tweets."
    )
    parser.add_argument("--step", choices=["train", "eval", "annotate"], default=None,
                        help="Passo a executar (default: train + eval + annotate)")
    parser.add_argument("--curated-csv", default=None,
                        help="CSV curado (default: data/curated_tweets_stance.csv)")
    parser.add_argument("--model-name", default=None,
                        help="Modelo base HuggingFace (default: BERTimbau)")
    parser.add_argument("--model-dir",  default=None,
                        help="Diretório para guardar/carregar o modelo fine-tuned")
    parser.add_argument("--tweets-dir", nargs="+", default=None,
                        help="Diretórios com CSVs de tweets raw (para anotação)")
    parser.add_argument("--annotation-output", default=None,
                        help="Caminho do CSV de pseudo-labels (saída da anotação)")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Treinar apenas as top-N camadas do encoder (0=full, default: 0)")
    parser.add_argument("--max-tweets", type=int, default=2_350_000)
    parser.add_argument("--irony-filter", action="store_true",
                        help="Remover tweets irônicos antes da anotação de stance")
    parser.add_argument("--irony-model-dir", default=None,
                        help="Diretório do modelo de ironia (default: models/bert_irony)")
    parser.add_argument("--irony-threshold", type=float, default=None,
                        help="Threshold para o filtro de ironia (default: 0.10)")
    args = parser.parse_args()

    cfg = BERTConfig(max_tweets=args.max_tweets)
    if args.curated_csv:
        cfg.curated_csv = Path(args.curated_csv)
    if args.model_name:
        cfg.model_name = args.model_name
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)
    if args.tweets_dir:
        cfg.raw_data_dirs = [Path(d) for d in args.tweets_dir]
    if args.annotation_output:
        cfg.annotation_output = Path(args.annotation_output)
    if args.freeze_layers > 0:
        cfg.freeze_layers = args.freeze_layers
    if args.irony_filter:
        cfg.irony_filter = True
    if args.irony_model_dir:
        cfg.irony_model_dir = Path(args.irony_model_dir)
    if args.irony_threshold is not None:
        cfg.irony_threshold = args.irony_threshold

    annotator = BERTStanceAnnotator(cfg)
    steps = [args.step] if args.step else ["train", "eval", "annotate"]
    for step in steps:
        print(f"\n{'='*60}\nPASSO: {step.upper()}\n{'='*60}")
        if step == "train":
            annotator.train()
        elif step == "eval":
            annotator.evaluate()
        elif step == "annotate":
            annotator.annotate()


if __name__ == "__main__":
    main()
