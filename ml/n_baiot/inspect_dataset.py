from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "raw"
)


def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    if not csv_files:
        print("No CSV files found yet.")
        return

    for path in csv_files:
        print("\n" + "=" * 80)
        print(f"File: {path.relative_to(PROJECT_ROOT)}")

        df = pd.read_csv(
            path,
            nrows=5,
        )

        print(f"Columns: {len(df.columns)}")
        print(f"Shape preview: {df.shape}")

        print("\nFirst columns:")
        for column in df.columns[:10]:
            print(f"  - {column}")

        print("\nFirst rows:")
        print(df.head())


if __name__ == "__main__":
    main()