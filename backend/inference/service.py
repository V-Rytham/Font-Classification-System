from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


@dataclass
class Prediction:
    font: str
    confidence: float


class FontClassifierService:
    """Loads a trained Keras model and performs font predictions."""

    def __init__(self, model_path: Path, classes_dir: Path, input_size: int = 100) -> None:
        self.model_path = model_path
        self.classes_dir = classes_dir
        self.input_size = input_size
        self.model: tf.keras.Model | None = None
        self.class_names: List[str] = []

    def _labels_from_json(self) -> List[str]:
        labels_path = self.model_path.with_name("class_names.json")
        if not labels_path.exists():
            return []

        with labels_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError(f"Invalid label file format: {labels_path}")

        return labels

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. Train the model first or provide MODEL_PATH."
            )

        json_labels = self._labels_from_json()
        if json_labels:
            self.class_names = json_labels
        else:
            if not self.classes_dir.exists():
                raise FileNotFoundError(
                    f"Classes directory not found: {self.classes_dir}."
                )

            self.class_names = sorted(
                [p.name for p in self.classes_dir.iterdir() if p.is_dir() and p.name != "LICENSE.txt"]
            )

        if not self.class_names:
            raise ValueError("No font class labels found.")

        self.model = tf.keras.models.load_model(str(self.model_path))

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = ImageOps.fit(
            image,
            (self.input_size, self.input_size),
            method=Image.Resampling.LANCZOS,
            bleed=0.0,
            centering=(0.5, 0.5),
        )

        arr = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Prediction]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        sample = self._prepare_image(image)
        probs = self.model.predict(sample, verbose=0)[0]

        top_k = max(1, min(top_k, len(self.class_names)))
        top_indices = np.argsort(probs)[-top_k:][::-1]

        return [
            Prediction(font=self.class_names[idx], confidence=float(probs[idx]))
            for idx in top_indices
        ]
