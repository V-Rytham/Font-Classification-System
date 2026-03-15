import argparse
import json
from pathlib import Path

import numpy as np

from utils import crop_dataset, list_dataset, load_dataset, build_model, IMG_HEIGHT, IMG_WIDTH


def parse_args():
    parser = argparse.ArgumentParser(description="Train the font classification CNN model.")
    parser.add_argument("--train-dir", type=Path, default=Path("train"), help="Path to training dataset root")
    parser.add_argument("--valid-dir", type=Path, default=Path("valid"), help="Path to validation dataset root")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=300, help="Training batch size")
    parser.add_argument("--valid-batch-size", type=int, default=100, help="Validation batch size")
    parser.add_argument("--output-model", type=Path, default=Path("CNN_Font_Classification.h5"), help="Output .h5 path")
    parser.add_argument("--output-labels", type=Path, default=Path("class_names.json"), help="Output class names JSON path")
    parser.add_argument("--crop-dataset", action="store_true", help="Crop all images to 100x100 before training")
    return parser.parse_args()


def validate_dir(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")


def main():
    args = parse_args()

    validate_dir(args.train_dir, "Train directory")
    validate_dir(args.valid_dir, "Validation directory")

    if args.crop_dataset:
        print("Cropping datasets to 100x100...")
        crop_dataset(str(args.train_dir))
        crop_dataset(str(args.valid_dir))

    train_data, image_count_train, class_names = list_dataset(str(args.train_dir))
    valid_data, image_count_valid, _ = list_dataset(str(args.valid_dir))

    if image_count_train == 0 or image_count_valid == 0:
        raise ValueError("Training/validation dataset appears empty. Check dataset paths.")

    output_classes = len(class_names)
    print(f"Classes: {output_classes}")
    print(f"Training images: {image_count_train}")
    print(f"Validation images: {image_count_valid}")

    steps_per_epoch = int(np.ceil(image_count_train / args.train_batch_size))

    train_data_gen = load_dataset(train_data, args.train_batch_size, IMG_HEIGHT, IMG_WIDTH, class_names)
    valid_data_gen = load_dataset(valid_data, args.valid_batch_size, IMG_HEIGHT, IMG_WIDTH, class_names)

    model = build_model(output_classes=output_classes, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(
        train_data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=valid_data_gen,
        validation_steps=max(1, image_count_valid // args.valid_batch_size),
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_model), overwrite=True)
    print(f"Saved model: {args.output_model}")

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    with args.output_labels.open("w", encoding="utf-8") as f:
        json.dump(class_names.tolist(), f, indent=2)
    print(f"Saved class labels: {args.output_labels}")


if __name__ == "__main__":
    main()
