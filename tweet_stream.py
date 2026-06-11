"""
Leitura eficiente de um grande corpus de tweets distribuído em vários CSVs.

RawTweetStream itera de forma lazy sobre todos os arquivos CSV de um ou mais
diretórios, aplicando deduplicação por conversation_id e pulando tweets
já anotados.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterator

import pandas as pd


class RawTweetStream:
    """Iterator lazy sobre CSVs de tweets.

    Lê os arquivos em chunks para não carregar tudo em memória.
    Filtra IDs já anotados em cada chunk antes de entregá-los ao chamador.
    """

    def __init__(
        self,
        data_dirs: list[Path],
        skip_ids: set[str],
        chunk_size: int = 10_000,
        max_tweets: int = 2_350_000,
    ):
        self.data_dirs = data_dirs
        self.skip_ids = skip_ids
        self.chunk_size = chunk_size
        self.max_tweets = max_tweets

    def __iter__(self) -> Iterator[pd.DataFrame]:
        total = 0
        for data_dir in self.data_dirs:
            csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
            for csv_file in csv_files:
                if total >= self.max_tweets:
                    return
                try:
                    chunks = pd.read_csv(
                        csv_file, dtype=str, chunksize=self.chunk_size,
                        usecols=["conversation_id", "text"],
                    )
                    for chunk in chunks:
                        if total >= self.max_tweets:
                            return
                        chunk = chunk.dropna(subset=["conversation_id", "text"])
                        chunk["conversation_id"] = chunk["conversation_id"].str.strip()
                        chunk["text"] = chunk["text"].str.strip()
                        chunk = chunk[chunk["text"] != ""]
                        chunk = chunk.drop_duplicates(subset=["conversation_id"])
                        chunk = chunk[~chunk["conversation_id"].isin(self.skip_ids)]
                        if len(chunk) > 0:
                            remaining = self.max_tweets - total
                            chunk = chunk.head(remaining)
                            total += len(chunk)
                            yield chunk
                except Exception as e:
                    print(f"Aviso: ignorando {csv_file}: {e}")
