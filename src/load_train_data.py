from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/train.csv")


def main() -> None:
    train_data = pd.read_csv(DATA_PATH)
    print(train_data.head())
    print(f"rows: {len(train_data)}")


if __name__ == "__main__":
    main()
