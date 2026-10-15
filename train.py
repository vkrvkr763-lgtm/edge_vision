
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"

# Lightweight pretrained YOLO11 model.
MODEL_NAME = "yolo11n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}\n"
            "Run prepare_dataset.py first."
        )

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("Training on CPU. This may be slow.")
        device = "cpu"

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=4,
        patience=10,
        project=str(BASE_DIR / "runs" / "detect"),
        name="edgevision",
        exist_ok=True,
        seed=42,
        plots=True,
        save=True,
        verbose=True
    )

    best_model = (
        BASE_DIR / "runs" / "detect" /
        "edgevision" / "weights" / "best.pt"
    )

    print("\nTraining finished.")
    print("Best checkpoint:", best_model)
    print("Checkpoint exists:", best_model.exists())

if __name__ == "__main__":
    main()
