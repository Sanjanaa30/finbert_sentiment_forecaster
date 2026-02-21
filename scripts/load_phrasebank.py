from pathlib import Path
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path("data/processed/phrasebank_allagree.csv")
DATASET_REPO = "takala/financial_phrasebank"
ZIP_FILENAME = "data/FinancialPhraseBank-v1.0.zip"
ZIP_MEMBER = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"


def parse_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Rows are stored as: sentence@label
        sentence, label = line.rsplit("@", 1)
        rows.append({"sentence": sentence.strip(), "label": label.strip()})

    return rows


def main() -> None:
    zip_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=ZIP_FILENAME,
    )

    with zipfile.ZipFile(zip_path) as zf:
        raw_text = zf.read(ZIP_MEMBER).decode("latin-1")

    df = pd.DataFrame(parse_rows(raw_text))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}  rows={len(df)}  cols={list(df.columns)}")


if __name__ == "__main__":
    main()
