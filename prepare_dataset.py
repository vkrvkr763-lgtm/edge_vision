
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Original dataset
BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "Pothole Detection Group.v6-yolo-v11.voc"

# YOLO dataset output
OUTPUT_DIR = BASE_DIR / "dataset"

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_xml_to_yolo(xml_path, label_path):
    """Convert Pascal VOC bounding boxes to YOLO format."""

    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {xml_path}")

    labels = []

    for obj in root.findall("object"):
        class_name = (
            obj.findtext("name", default="").strip().lower()
        )

        # This project detects potholes only.
        if class_name != "pothole":
            continue

        box = obj.find("bndbox")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Clip coordinates to image boundaries.
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            print(f"Skipping invalid box in {xml_path.name}")
            continue

        # YOLO: class_id x_center y_center box_width box_height
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        labels.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    # Empty files are valid for images with no potholes.
    label_path.write_text("\n".join(labels))


def prepare_split(source_split, output_split):
    source_folder = SOURCE_DIR / source_split

    if not source_folder.is_dir():
        raise FileNotFoundError(
            f"Dataset split not found: {source_folder}"
        )

    image_output = OUTPUT_DIR / "images" / output_split
    label_output = OUTPUT_DIR / "labels" / output_split

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0
    missing_xml = 0

    for image_path in sorted(source_folder.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        xml_path = image_path.with_suffix(".xml")

        if not xml_path.is_file():
            print(f"Warning: missing XML for {image_path.name}")
            missing_xml += 1
            continue

        destination_image = image_output / image_path.name
        destination_label = (
            label_output / f"{image_path.stem}.txt"
        )

        # Copy image without modifying the original dataset.
        shutil.copy2(image_path, destination_image)

        convert_xml_to_yolo(xml_path, destination_label)

        label_lines = [
            line for line in destination_label.read_text().splitlines()
            if line.strip()
        ]

        image_count += 1
        object_count += len(label_lines)

    print(
        f"{output_split}: {image_count} images, "
        f"{object_count} pothole boxes, "
        f"{missing_xml} images skipped for missing XML"
    )


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Original dataset folder not found: {SOURCE_DIR}"
        )

    for source_split, output_split in SPLITS.items():
        prepare_split(source_split, output_split)

    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_text = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
"""

    yaml_path.write_text(yaml_text)

    print("\nDataset preparation complete!")
    print(f"YOLO configuration: {yaml_path}")
    print(f"Dataset location: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()