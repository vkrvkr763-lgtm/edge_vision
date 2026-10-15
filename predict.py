
from pathlib import Path
import argparse
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR / "runs" / "detect" /
    "edgevision" / "weights" / "best.pt"
)

def main():
    parser = argparse.ArgumentParser(
        description="EdgeVision pothole detection"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the road image"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence"
    )
    args = parser.parse_args()

    image_path = Path(args.source).expanduser().resolve()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first."
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        imgsz=640,
        save=True,
        project=str(BASE_DIR / "runs" / "predict"),
        name="results",
        exist_ok=True
    )

    for result in results:
        print(f"\nImage: {image_path.name}")
        print(f"Potholes detected: {len(result.boxes)}")

        for i, box in enumerate(result.boxes, start=1):
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            print(
                f"Detection {i}: confidence={confidence:.3f}, "
                f"box={coordinates}"
            )

    print("\nAnnotated image saved under:")
    print(BASE_DIR / "runs" / "predict" / "results")

if __name__ == "__main__":
    main()
