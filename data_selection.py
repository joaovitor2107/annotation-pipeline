"""
Curação do conjunto de treino a partir das anotações manuais.

Lê o CSV de anotações humanas, filtra apenas as conversas sem ambiguidade
(todos os anotadores concordam), balanceia por classe, junta o texto dos tweets
dos CSVs raw, e gera o split train/val/test estratificado.

Saída: curated_tweets_stance.csv com coluna 'split'.

Uso:
    python data_selection.py \
        --annotations data/anotacoes.csv \
        --tweets-dir data/tweets/ \
        --output data/curated_tweets_stance.csv
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# Labels canónicos usados em todo o pipeline
STANCE_LABELS = ["a favor", "contra", "neutro"]

# Mapeamento das anotações em inglês para português
POSITION_MAPPING = {
    "in favor": "a favor",
    "against":  "contra",
    "neutral":  "neutro",
}


@dataclass
class SelectionConfig:
    annotation_csv: Path = Path("data/anotacoes.csv")
    raw_data_dirs: list[Path] = field(default_factory=list)
    output_path: Path = Path("data/curated_tweets_stance.csv")
    max_per_class: int = 500
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 27
    oversample_train: bool = False


class DataSelector:
    """Cura e divide o conjunto de anotações manuais."""

    def __init__(self, cfg: SelectionConfig):
        self.cfg = cfg
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        df = self._load_annotations()
        df = self._normalise_labels(df)
        df = self._filter_unambiguous(df)
        df = self._balance_classes(df)
        df = self._join_tweet_text(df)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]
        df = self._add_split_column(df)
        if self.cfg.oversample_train:
            df = self._oversample_train(df)

        df.to_csv(self.cfg.output_path, index=False)
        self._print_summary(df)
        print(f"\nGuardado em: {self.cfg.output_path}")
        return self.cfg.output_path

    def _load_annotations(self) -> pd.DataFrame:
        # Tenta separador ";" (comum em exports do Excel) e "," como fallback
        try:
            df = pd.read_csv(self.cfg.annotation_csv, dtype=str, sep=";")
            if len(df.columns) < 2:
                raise ValueError
        except Exception:
            df = pd.read_csv(self.cfg.annotation_csv, dtype=str)
        df.columns = df.columns.str.strip()
        for col in ["conversation_id", "Posicao_Final"]:
            df[col] = df[col].str.strip()
        return df[["conversation_id", "Posicao_Final"]].copy()

    def _normalise_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = df["Posicao_Final"].str.lower().map(POSITION_MAPPING)
        df = df.dropna(subset=["label"])
        return df

    def _filter_unambiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby("conversation_id")["label"].nunique()
        unambiguous_ids = counts[counts == 1].index
        filtered = df[df["conversation_id"].isin(unambiguous_ids)]
        filtered = filtered.drop_duplicates(subset=["conversation_id"])
        removed = len(df["conversation_id"].unique()) - len(filtered)
        print(f"Removidas {removed} conversas ambíguas; restam {len(filtered)}")
        return filtered

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for label in STANCE_LABELS:
            subset = df[df["label"] == label]
            n = min(len(subset), self.cfg.max_per_class)
            frames.append(subset.sample(n=n, random_state=self.cfg.seed))
        balanced = pd.concat(frames, ignore_index=True)
        print(f"Distribuição após balanceamento: {Counter(balanced['label'])}")
        return balanced

    def _join_tweet_text(self, df: pd.DataFrame) -> pd.DataFrame:
        text_map: dict[str, str] = {}
        target_ids = set(df["conversation_id"].values)
        for data_dir in self.cfg.raw_data_dirs:
            for csv_file in glob.glob(str(data_dir / "*.csv")):
                try:
                    for chunk in pd.read_csv(csv_file, dtype=str, chunksize=50_000,
                                              usecols=["conversation_id", "text"]):
                        chunk = chunk.dropna(subset=["conversation_id", "text"])
                        chunk["conversation_id"] = chunk["conversation_id"].str.strip()
                        for cid, group in chunk.groupby("conversation_id"):
                            if cid in target_ids and cid not in text_map:
                                texts = group["text"].dropna().str.strip().tolist()
                                text_map[cid] = " | ".join(texts)
                except Exception as e:
                    print(f"Aviso: {csv_file}: {e}")
        df = df.copy()
        df["text"] = df["conversation_id"].map(text_map)
        missing = df["text"].isna().sum()
        if missing:
            print(f"Aviso: {missing} conversas sem texto nos CSVs raw")
        return df

    def _add_split_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df["split"] = "test"
        val_size = self.cfg.val_ratio
        test_size = 1.0 - self.cfg.train_ratio - val_size

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size,
                                       random_state=self.cfg.seed)
        train_idx, rest_idx = next(sss1.split(df, df["label"]))
        df.loc[train_idx, "split"] = "train"

        df_rest = df.iloc[rest_idx].reset_index(drop=True)
        relative_val = val_size / (val_size + test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - relative_val,
                                       random_state=self.cfg.seed)
        val_rel_idx, _ = next(sss2.split(df_rest, df_rest["label"]))
        val_original_idx = [rest_idx[i] for i in val_rel_idx]
        df.loc[val_original_idx, "split"] = "val"
        return df

    def _oversample_train(self, df: pd.DataFrame) -> pd.DataFrame:
        train_mask = df["split"] == "train"
        df_train = df[train_mask].copy()
        df_other = df[~train_mask].copy()
        max_count = Counter(df_train["label"]).most_common(1)[0][1]
        frames = [df_train]
        for label, count in Counter(df_train["label"]).items():
            if count < max_count:
                extra = df_train[df_train["label"] == label].sample(
                    n=max_count - count, replace=True, random_state=self.cfg.seed)
                frames.append(extra)
        df_train_os = pd.concat(frames, ignore_index=True)
        print(f"Treino após oversampling: {Counter(df_train_os['label'])}")
        return pd.concat([df_train_os, df_other], ignore_index=True)

    def _print_summary(self, df: pd.DataFrame):
        print("\n=== Dataset Curado ===")
        for split in ["train", "val", "test"]:
            subset = df[df["split"] == split]
            dist = Counter(subset["label"])
            print(f"  {split:5s}: {len(subset):4d} | {dict(dist)}")
        print(f"  {'total':5s}: {len(df):4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="CSV com anotações manuais (colunas: conversation_id, Posicao_Final)")
    parser.add_argument("--tweets-dir", nargs="+", required=True,
                        help="Um ou mais diretórios com CSVs de tweets raw")
    parser.add_argument("--output", default="data/curated_tweets_stance.csv")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--oversample", action="store_true")
    args = parser.parse_args()

    cfg = SelectionConfig(
        annotation_csv=Path(args.annotations),
        raw_data_dirs=[Path(d) for d in args.tweets_dir],
        output_path=Path(args.output),
        max_per_class=args.max_per_class,
        oversample_train=args.oversample,
    )
    DataSelector(cfg).run()
