from typing import Generator
import os
import csv

import loguru

logger = loguru.logger


def load_parallel_sentences(
    directory: str, lang1: str = "en", lang2: str = "fr", num_samples: int = 100000
) -> Generator[tuple[str, str], None, None]:
    """
    Reads parallel text files from a directory and yields sentence pairs as tuples.

    Args:
        directory (str): Path to the directory containing WMT data.
        lang1 (str): Source language suffix (default: "en").
        lang2 (str): Target language suffix (default: "fr").

    Yields:
        tuple: (sentence_lang1, sentence_lang2)
    """
    file1 = os.path.join(directory, f"giga-fren.release2.fixed.{lang1}")
    file2 = os.path.join(directory, f"giga-fren.release2.fixed.{lang2}")

    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError(f"Expected files {file1} and {file2} not found.")

    logger.info(f"loading {num_samples} samples from WMT English to French dataset.")
    with open(file1, "r", encoding="utf-8") as f1, open(
        file2, "r", encoding="utf-8"
    ) as f2:
        i = 0
        for line1, line2 in zip(f1, f2):
            if i == num_samples:
                logger.info(f"finished looading {num_samples} dataset samples")
                break
            yield line1.strip(), line2.strip()
            i += 1


KAGGLE_CSV = "wmt14_translate_fr-en_{stage}.csv"


def load_kaggle_format(
    directory: str, stage: str, start_index: int = 0, num_samples: int | None = None
):
    file_path = os.path.join(directory, KAGGLE_CSV.format(stage=stage))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file {file_path} not found.")

    logger.info(f"loading {num_samples if num_samples else 'All'} samples from WMT English to French dataset stage: {stage}.")
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=["en", "fr"])
        for _ in range(start_index):
            reader.__next__()

        for i, row in enumerate(reader):
            if num_samples and i >= num_samples:
                break
            yield row["en"], row["fr"]


def kaggle_dataset_len(directory: str, stage: str) -> int:
    file_path = os.path.join(directory, KAGGLE_CSV.format(stage))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file {file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=["en", "fr"])
        return len(list(reader))
