"""
Calibração de thresholds de confiança por classe.

Um threshold de confiança uniforme retém demasiados 'neutro' ruidosos e
descarta demasiados 'a favor' nos pseudo-labels. Este script escolhe
thresholds por classe via calibração de precisão no test split humano:
para cada classe c, o menor threshold t_c na grelha tal que
precision(pred=c, conf>=t_c) atinge o alvo.

Por defeito, o test split é restrito a tweets não-irônicos, para coincidir
com a pipeline de anotação com --irony-filter (estratégia "ignorar").

Uso:
    python threshold_calibration.py \
        --model-dir models/bert_stance \
        --pseudo-csv data/pseudo_labeled_stance.csv \
        --target-precision 0.90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BertForSequenceClassification

from bert_annotator import (
    BERTConfig,
    StanceDataset,
    LABEL2ID,
    _get_device,
)
from data_selection import STANCE_LABELS

GRID = [0.90, 0.95, 0.97, 0.99, 0.995]


def predict_test_split(model_dir: Path, curated_csv: Path, drop_ironic: bool):
    """Devolve (labels_verdadeiros, labels_preditos, confianças) no test split humano."""
    cfg = BERTConfig()
    df = pd.read_csv(curated_csv, dtype=str)
    df = df[df["label"].isin(STANCE_LABELS)].copy()
    test_df = df[df["split"] == "test"]
    if drop_ironic and "irony" in test_df.columns:
        before = len(test_df)
        test_df = test_df[test_df["irony"] != "ironic"]
        print(f"Removidos {before - len(test_df)} tweets irônicos do test split "
              f"({len(test_df)} restantes) — coincide com a pipeline --irony-filter")

    device = _get_device(infer=True)  # CPU — bug de inferência BERT no MPS
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, normalization=True)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    label_ids = test_df["label"].map(LABEL2ID).tolist()
    dataset = StanceDataset(test_df["text"].tolist(), label_ids, tokenizer, cfg.max_length)
    loader = DataLoader(dataset, batch_size=cfg.infer_batch_size)

    preds, confs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.cpu().tolist())
            confs.extend(conf.cpu().tolist())

    true = np.array(label_ids)
    pred = np.array(preds)
    conf = np.array(confs)
    return true, pred, conf


def calibrate(true, pred, conf, target_precision: float) -> dict:
    """Menor threshold da grelha, por classe, que atinge a precisão alvo."""
    thresholds = {}
    for cls, cls_id in LABEL2ID.items():
        print(f"\n{cls}:")
        chosen = None
        for t in GRID:
            mask = (pred == cls_id) & (conf >= t)
            n = int(mask.sum())
            if n == 0:
                print(f"  t={t:5.3f}  n=0 — nenhuma predição sobrevive")
                continue
            precision = float((true[mask] == cls_id).mean())
            print(f"  t={t:5.3f}  n={n:4d}  precisão={precision:.3f}")
            if chosen is None and precision >= target_precision:
                chosen = t
        if chosen is None:
            chosen = GRID[-1]
            print(f"  AVISO: precisão alvo {target_precision} nunca atingida; "
                  f"a usar o threshold máximo da grelha {chosen}")
        thresholds[cls] = chosen
        print(f"  → threshold[{cls}] = {chosen}")
    return thresholds


def project_distribution(pseudo_csv: Path, thresholds: dict, fallback: float) -> dict:
    """Distribuição de classes do pool de pseudo-labels após o filtro por classe."""
    df = pd.read_csv(pseudo_csv, dtype=str)
    df["confidence"] = df["confidence"].astype(float)
    thr = df["predicted_label"].map(lambda c: thresholds.get(c, fallback))
    kept = df[df["confidence"] >= thr]
    dist = kept["predicted_label"].value_counts()
    print(f"\nPool de pseudo-labels projetado após filtro por classe "
          f"({len(kept)}/{len(df)} = {len(kept)/len(df):.1%} retidos):")
    for cls in STANCE_LABELS:
        n = int(dist.get(cls, 0))
        print(f"  {cls:10s}: {n:7d}  ({n / max(len(kept), 1):.1%})")
    return {
        "total_antes": len(df),
        "total_apos": len(kept),
        "distribuicao": {cls: int(dist.get(cls, 0)) for cls in STANCE_LABELS},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibra thresholds de confiança por classe")
    parser.add_argument("--model-dir", default="models/bert_stance")
    parser.add_argument("--curated-csv", default="data/curated_tweets_stance.csv")
    parser.add_argument("--pseudo-csv", default="data/pseudo_labeled_stance.csv")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--keep-ironic", action="store_true",
                        help="Mantém tweets irônicos no test split de calibração")
    parser.add_argument("--output", default="results/threshold_calibration.json")
    args = parser.parse_args()

    true, pred, conf = predict_test_split(
        Path(args.model_dir), Path(args.curated_csv), drop_ironic=not args.keep_ironic
    )
    print(f"\nTest split: {len(true)} exemplos")
    print("Distribuição dos labels humanos (referência para sanity check):")
    for cls, cls_id in LABEL2ID.items():
        n = int((true == cls_id).sum())
        print(f"  {cls:10s}: {n:4d}  ({n / len(true):.1%})")

    thresholds = calibrate(true, pred, conf, args.target_precision)
    projection = project_distribution(Path(args.pseudo_csv), thresholds, fallback=max(GRID))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "target_precision": args.target_precision,
            "grid": GRID,
            "thresholds": thresholds,
            "test_split_size": len(true),
            "projection": projection,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {out}")
    cli = ",".join(f"{c}={t}" for c, t in thresholds.items())
    print(f"\nPróximo passo:\n  python confidence_filter.py \\\n"
          f"      --input {args.pseudo_csv} \\\n"
          f"      --threshold-per-class \"{cli}\" \\\n"
          f"      --output data/pseudo_labeled_filtered_perclass.csv")


if __name__ == "__main__":
    main()
