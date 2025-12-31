from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "mvtec" / "bottle"

train_good = DATA_PATH / "train" / "good"
test_path = DATA_PATH / "test"
gt_path  = DATA_PATH / "ground_truth"


print("Train images:", len(list(train_good.glob("*.png"))))
