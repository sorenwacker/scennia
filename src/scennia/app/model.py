import json
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL.Image import Image
from torchvision import transforms

from scennia.app.data import CellPrediction


class ModelManager:
    """Manages ONNX model loading and classification without global variables"""

    def __init__(self):
        self.onnx_model_path = None
        self.onnx_classification_model = None
        self.onnx_model_metadata = None
        self.onnx_transform_for_classification = None
        self.cellpose_model = None

    def set_onnx_model_path(self, model_path):
        self.onnx_model_path = model_path

    def load_onnx_model_if_needed(self):
        """Load ONNX classification model and metadata"""
        if not self.onnx_model_path or self.is_onnx_model_loaded():
            return False

        try:
            # Load ONNX model
            self.onnx_classification_model = ort.InferenceSession(self.onnx_model_path)
            print(f"Loaded classification model from {self.onnx_model_path}")

            # Load metadata
            metadata_path = os.path.join(os.path.dirname(self.onnx_model_path), "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.onnx_model_metadata = json.load(f)
                print(f"Loaded model metadata: {self.onnx_model_metadata['num_classes']} classes")
                print(f"Classes: {self.onnx_model_metadata['class_names']}")
            else:
                print("Warning: No metadata file found")
                self.onnx_model_metadata = None

            # Setup transforms for classification
            img_size = self.onnx_model_metadata.get("img_size", 224) if self.onnx_model_metadata else 224
            mean = (
                self.onnx_model_metadata.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
                if self.onnx_model_metadata
                else [0.485, 0.456, 0.406]
            )
            std = (
                self.onnx_model_metadata.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
                if self.onnx_model_metadata
                else [0.229, 0.224, 0.225]
            )

            self.onnx_transform_for_classification = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            return True
        except Exception as e:
            print(f"Error loading classification model: {e}")
            return False

    def classify_cell_crop(self, cropped_image: Image) -> CellPrediction:
        """Classify a cell crop using the loaded ONNX model"""
        if self.onnx_classification_model is None or self.onnx_transform_for_classification is None:
            msg = "Classification model not loaded"
            raise Exception(msg)

        # Ensure image is RGB
        if cropped_image.mode != "RGB":
            cropped_image = cropped_image.convert("RGB")

        # Apply transforms
        input_tensor = self.onnx_transform_for_classification(cropped_image).unsqueeze(0).numpy()  # type: ignore[]

        # Run inference
        inputs = {self.onnx_classification_model.get_inputs()[0].name: input_tensor}
        outputs = self.onnx_classification_model.run(None, inputs)

        # Convert outputs to probabilities
        predictions = F.softmax(torch.tensor(outputs[0][0]), dim=0).numpy()

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Get class name if metadata available - Combined if statement (fixes SIM102)
        class_name = "Unknown"
        if (
            self.onnx_model_metadata
            and "class_names" in self.onnx_model_metadata
            and predicted_class_idx < len(self.onnx_model_metadata["class_names"])
        ):
            class_name = self.onnx_model_metadata["class_names"][predicted_class_idx]

        # Parse treatment and concentration from class name
        treatment_type = "unknown"
        concentration = 0

        if "_" in class_name:
            parts = class_name.split("_")
            if len(parts) >= 2:
                treatment_type = parts[0]
                try:
                    concentration = int(parts[1])
                except ValueError:
                    concentration = 0
        return CellPrediction(
            predicted_class=class_name,
            predicted_class_idx=int(predicted_class_idx),
            confidence=confidence,
            treatment_type=treatment_type,
            concentration=concentration,
            all_predictions=predictions.tolist(),
        )

    def has_onnx_model_path(self):
        """Check if ONNX model path has been set"""
        return self.onnx_model_path is not None

    def is_onnx_model_loaded(self):
        """Check if ONNX model is loaded and ready"""
        return self.onnx_classification_model is not None and self.onnx_transform_for_classification is not None

    def get_cellpose_model(self):
        from cellpose import models

        if self.cellpose_model:
            return self.cellpose_model
        # Load cellpose model
        self.cellpose_model = models.CellposeModel(gpu=True)
        return self.cellpose_model
