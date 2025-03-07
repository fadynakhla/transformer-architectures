import os
from typing import Generator

def load_parallel_sentences(directory: str, lang1: str = "en", lang2: str = "fr") -> Generator[tuple[str, str], None, None]:
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

    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        i = 0
        for line1, line2 in zip(f1, f2):
            if i == 200000:
                break
            yield line1.strip(), line2.strip()
            i += 1
