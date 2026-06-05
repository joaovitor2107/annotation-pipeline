"""
Filtragem dos pseudo-labels por limiar de confiança.

Após a anotação automática, cada tweet tem um score de confiança
(softmax do logit mais alto). Este script retém apenas os tweets
acima do limiar definido, reduzindo o ruído no conjunto de pseudo-labels.

Uso:
    python confidence_filter.py \
        --input  data/pseudo_labeled_stance.csv \
        --output data/pseudo_labeled_filtered_0.97.csv \
        --threshold 0.97
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_selection import STANCE_LABELS


@dataclass
class FilterConfig:
    input_csv: Path = Path("data/pseudo_labeled_stance.csv")
    output_csv: Path = Path("data/pseudo_labeled_filtered_stance.csv")
    confidence_threshold: float = 0.97
    min_per_class: int = 100


class ConfidenceFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        df = pd.read_csv(self.cfg.input_csv, dtype=str)
        df["confidence"] = df["confidence"].astype(float)

        self._print_stats(df, "Antes do filtro")
        df_filtered = df[df["confidence"] >= self.cfg.confidence_threshold].copy()
        self._print_stats(df_filtered, f"Após filtro (conf ≥ {self.cfg.confidence_threshold})")

        self._warn_if_low(df_filtered)

        df_filtered.to_csv(self.cfg.output_csv, index=False)

        summary = {
            "total_antes": len(df),
            "total_apos": len(df_filtered),
            "taxa_retencao": round(len(df_filtered) / max(len(df), 1), 4),
            "threshold": self.cfg.confidence_threshold,
            "distribuicao": dict(Counter(df_filtered["predicted_label"])),
        }
        summary_path = self.cfg.output_csv.parent / "filter_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nFicheiro filtrado: {self.cfg.output_csv}")
        print(f"Resumo: {self.cfg.output_csv.parent / 'filter_summary.json'}")
        return self.cfg.output_csv

    def _print_stats(self, df: pd.DataFrame, label: str):
        dist = Counter(df["predicted_label"])
        print(f"\n{label}:")
        print(f"  Total: {len(df):,}")
        for cls in STANCE_LABELS:
            print(f"  {cls:10s}: {dist.get(cls, 0):,}")

    def _warn_if_low(self, df: pd.DataFrame):
        dist = Counter(df["predicted_label"])
        for cls in STANCE_LABELS:
            n = dist.get(cls, 0)
            if n < self.cfg.min_per_class:
                print(f"AVISO: classe '{cls}' tem apenas {n} exemplos. "
                      f"Considere diminuir o threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,
                        help="CSV de pseudo-labels (saída de bert_annotator.py)")
    parser.add_argument("--output",    required=True,
                        help="Caminho do CSV filtrado")
    parser.add_argument("--threshold", type=float, default=0.97,
                        help="Confiança mínima para reter um tweet (default: 0.97)")
    args = parser.parse_args()

    cfg = FilterConfig(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        confidence_threshold=args.threshold,
    )
    ConfidenceFilter(cfg).run()
