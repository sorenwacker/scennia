import json
import os

import numpy as np
import onnxruntime as ort
from torchvision import transforms


class ModelManager:
    """Manages ONNX model loading and classification without global variables"""

    def __init__(self):
        self.classification_model = None
        self.model_metadata = None
        self.transform_for_classification = None

    def load_model(self, model_path):
        """Load ONNX classification model and metadata"""
        try:
            # Load ONNX model
            self.classification_model = ort.InferenceSession(model_path)
            print(f"Loaded classification model from {model_path}")

            # Load metadata
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                print(f"Loaded model metadata: {self.model_metadata['num_classes']} classes")
                print(f"Classes: {self.model_metadata['class_names']}")
            else:
                print("Warning: No metadata file found")
                self.model_metadata = None

            # Setup transforms for classification
            img_size = self.model_metadata.get("img_size", 224) if self.model_metadata else 224
            mean = (
                self.model_metadata.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
                if self.model_metadata
                else [0.485, 0.456, 0.406]
            )
            std = (
                self.model_metadata.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
                if self.model_metadata
                else [0.229, 0.224, 0.225]
            )

            self.transform_for_classification = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            return True

        except Exception as e:
            print(f"Error loading classification model: {e}")
            return False

    def classify_cell_crop(self, cell_crop_pil):
        """Classify a cell crop using the loaded ONNX model"""
        if self.classification_model is None or self.transform_for_classification is None:
            return {"error": "Classification model not loaded"}

        try:
            # Ensure image is RGB
            if cell_crop_pil.mode != "RGB":
                cell_crop_pil = cell_crop_pil.convert("RGB")

            # Apply transforms
            input_tensor = self.transform_for_classification(cell_crop_pil).unsqueeze(0).numpy()

            # Run inference
            inputs = {self.classification_model.get_inputs()[0].name: input_tensor}
            outputs = self.classification_model.run(None, inputs)
            predictions = outputs[0][0]  # Get first batch item

            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))

            # Get class name if metadata available - Combined if statement (fixes SIM102)
            class_name = "Unknown"
            if (
                self.model_metadata
                and "class_names" in self.model_metadata
                and predicted_class_idx < len(self.model_metadata["class_names"])
            ):
                class_name = self.model_metadata["class_names"][predicted_class_idx]

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

            return {
                "predicted_class": class_name,
                "predicted_class_idx": int(predicted_class_idx),
                "confidence": confidence,
                "treatment_type": treatment_type,
                "concentration": concentration,
                "all_predictions": predictions.tolist(),
            }

        except Exception as e:
            print(f"Error in cell classification: {e}")
            return {"error": str(e)}

    def is_loaded(self):
        """Check if model is loaded and ready"""
        return self.classification_model is not None and self.transform_for_classification is not None


# Create global instance to replace global variables
model_manager = ModelManager()


def load_classification_model(model_path):
    """Load ONNX classification model and metadata"""
    return model_manager.load_model(model_path)


def classify_cell_crop(cell_crop_pil):
    """Classify a cell crop using the loaded ONNX model"""
    return model_manager.classify_cell_crop(cell_crop_pil)
