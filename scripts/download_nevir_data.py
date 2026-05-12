from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    dataset = load_dataset("orionweller/NevIR")

    for split in ["train", "validation", "test"]:
        file_path = DATA_DIR / f"{split}.csv"
        dataset[split].to_csv(file_path)
        print(f"saved {file_path} ({len(dataset[split])} rows)")


if __name__ == "__main__":
    main()
