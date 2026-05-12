from pathlib import Path

from datasets import load_dataset


def main() -> None:
    output_dir = Path("data/raw/nevir")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("orionweller/NevIR")

    for split_name, split_data in dataset.items():
        csv_path = output_dir / f"{split_name}.csv"
        jsonl_path = output_dir / f"{split_name}.jsonl"

        split_data.to_csv(csv_path)
        split_data.to_json(jsonl_path)

        print(f"saved {split_name}: {len(split_data)} rows")
        print(f"  csv:   {csv_path}")
        print(f"  jsonl: {jsonl_path}")


if __name__ == "__main__":
    main()
