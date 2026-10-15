
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
MODEL_PATH = (
    BASE_DIR / "runs" / "detect" /
    "edgevision" / "weights" / "best.pt"
)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        plots=True,
        project=str(BASE_DIR / "runs" / "evaluate"),
        name="test_results",
        exist_ok=True
    )

    print("\n--- EdgeVision Test Metrics ---")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@0.50:   {metrics.box.map50:.4f}")
    print(f"mAP@0.50:95:{metrics.box.map:.4f}")

    print("\nDetailed results saved under:")
    print(BASE_DIR / "runs" / "evaluate" / "test_results")

if __name__ == "__main__":
    main()
