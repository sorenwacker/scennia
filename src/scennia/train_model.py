import argparse
import json
import os
import time

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score
from torchvision import models, transforms


class HistogramEqualization:
    """Apply histogram equalization to normalize image intensities"""

    def __call__(self, img):
        # img is PIL Image
        if img.mode == "L":  # Grayscale
            # Convert to tensor for processing
            tensor = transforms.functional.to_tensor(img)
            tensor = self.equalize_tensor(tensor)
            return transforms.functional.to_pil_image(tensor)
        if img.mode == "RGB":
            # Apply to each channel separately
            r, g, b = img.split()
            r_eq = transforms.functional.to_pil_image(self.equalize_tensor(transforms.functional.to_tensor(r)))
            g_eq = transforms.functional.to_pil_image(self.equalize_tensor(transforms.functional.to_tensor(g)))
            b_eq = transforms.functional.to_pil_image(self.equalize_tensor(transforms.functional.to_tensor(b)))
            return Image.merge("RGB", (r_eq, g_eq, b_eq))
        return img

    def equalize_tensor(self, tensor):
        """Apply histogram equalization to a tensor"""
        # Flatten and get histogram
        flat = tensor.flatten()
        hist = torch.histc(flat, bins=256, min=0, max=1)

        # Compute cumulative distribution
        cdf = torch.cumsum(hist, dim=0)
        cdf = cdf / cdf[-1]  # Normalize

        # Map values
        indices = (flat * 255).long().clamp(0, 255)
        return cdf[indices].reshape(tensor.shape)


class AspectRatioResize:
    """Resize image while preserving aspect ratio and pad to square"""

    def __init__(self, target_size, fill_color=0):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img):
        # Get original dimensions
        if hasattr(img, "size"):  # PIL Image
            w, h = img.size
        else:  # torch tensor
            h, w = img.shape[-2:]

        # Calculate scale factor (resize so largest dimension fits)
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resize_transform = transforms.Resize((new_h, new_w), antialias=True)
        resized_img = resize_transform(img)

        # Calculate padding
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        pad_w_extra = (self.target_size - new_w) % 2
        pad_h_extra = (self.target_size - new_h) % 2

        # Apply padding (left, top, right, bottom)
        padding = (pad_w, pad_h, pad_w + pad_w_extra, pad_h + pad_h_extra)
        pad_transform = transforms.Pad(padding, fill=self.fill_color, padding_mode="constant")

        return pad_transform(resized_img)


class CellDataModule(L.LightningDataModule):
    def __init__(self, csv_path, img_size=224, batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        # Get parent directory of CSV file (one level up from processed/)
        self.csv_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
        self.img_size = img_size
        self.batch_size = batch_size
        self._setup_done = False

        # Transforms
        self.train_transforms = transforms.Compose(
            [
                # AspectRatioResize(img_size),
                transforms.Resize((img_size, img_size)),
                # HistogramEqualization(),
                # transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                # AspectRatioResize(img_size),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):  # noqa: ARG002
        if self._setup_done:
            return

        df = pd.read_csv(self.csv_path)

        # Create class labels
        def create_ordinal_class_name(row):
            treatment_type = row["treatment_type"]
            concentration = int(row["concentration"])
            padded_conc = f"{concentration:02d}"
            return f"{treatment_type}_{padded_conc}"

        df["class"] = df.apply(create_ordinal_class_name, axis=1)
        unique_classes = sorted(df["class"].unique())
        class_to_label = {class_name: i for i, class_name in enumerate(unique_classes)}
        df["label"] = df["class"].map(class_to_label)

        self.num_classes = len(unique_classes)
        self.class_names = unique_classes

        # Get source image statistics with MORE detailed balancing
        source_image_stats = (
            df.groupby("source_image")
            .agg({"label": ["first", "count"], "treatment_type": "first", "concentration": "first"})
            .reset_index()
        )

        source_image_stats.columns = ["source_image", "label", "cell_count", "treatment_type", "concentration"]

        # Create a composite stratification key for better balance
        source_image_stats["stratify_key"] = (
            source_image_stats["treatment_type"].astype(str) + "_" + source_image_stats["concentration"].astype(str)
        )

        print("Source image distribution:")
        print(source_image_stats.groupby("stratify_key")["source_image"].count())

        # First split: train vs (val + test)
        train_sources, temp_sources = train_test_split(
            source_image_stats["source_image"],
            test_size=0.3,
            stratify=source_image_stats["stratify_key"],
            random_state=42,
        )

        # Second split: val vs test
        temp_stats = source_image_stats[source_image_stats["source_image"].isin(temp_sources)]
        val_sources, test_sources = train_test_split(
            temp_stats["source_image"],
            test_size=0.5,
            stratify=temp_stats["stratify_key"],
            random_state=42,
        )

        # Create dataframes
        train_df = df[df["source_image"].isin(train_sources)].copy()
        val_df = df[df["source_image"].isin(val_sources)].copy()
        test_df = df[df["source_image"].isin(test_sources)].copy()

        # Create datasets with enhanced augmentation for training
        self.train_dataset = CellDataset(train_df, self.train_transforms, self.csv_dir)
        self.val_dataset = CellDataset(val_df, self.val_transforms, self.csv_dir)
        self.test_dataset = CellDataset(test_df, self.val_transforms, self.csv_dir)

        # Store for analysis
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        print("\nSplit statistics:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"\n{split_name}:")
            for class_name in unique_classes:
                class_data = split_df[split_df["class"] == class_name]
                n_images = class_data["source_image"].nunique() if len(class_data) > 0 else 0
                n_cells = len(class_data)
                print(f"  {class_name}: {n_images} images, {n_cells} cells")
            print(f"  Total cells: {len(split_df)}")

        # Calculate class weights based on training data
        class_weights = compute_class_weight("balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
        self.class_weights = torch.FloatTensor(class_weights)
        print(f"\nClass weights: {dict(zip(self.class_names, class_weights, strict=False))}")

        self._setup_done = True

    def get_example_images(self, n_per_class=2):
        """Get example images from each class for logging"""
        example_images = []

        for class_name in self.class_names:
            class_label = self.class_names.index(class_name)
            class_samples = self.test_df[self.test_df["label"] == class_label].sample(
                n=min(n_per_class, len(self.test_df[self.test_df["label"] == class_label]))
            )

            for _, row in class_samples.iterrows():
                img_path = row["cell_path"]

                # Handle relative paths by making them relative to CSV directory
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.csv_dir, img_path)

                try:
                    image = Image.open(img_path)

                    # Convert to RGB if needed
                    if image.mode == "L" or image.mode == "RGBA":
                        image = image.convert("RGB")

                    example_images.append(
                        {"image": image, "class_name": class_name, "label": class_label, "path": img_path}
                    )
                except FileNotFoundError:
                    print(f"Warning: Could not find image at {img_path}")
                    continue

        return example_images

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class CellDataset(Dataset):
    def __init__(self, df, transforms=None, csv_dir=None):
        self.df = df
        self.transforms = transforms
        self.csv_dir = csv_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row["cell_path"]

        # Handle relative paths by making them relative to CSV directory
        if self.csv_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.csv_dir, img_path)

        image = Image.open(img_path)

        # Convert grayscale to RGB if needed (most models expect 3 channels)
        if image.mode == "L":  # Grayscale
            image = image.convert("RGB")  # Convert to 3-channel RGB
        elif image.mode == "RGBA":  # RGBA
            image = image.convert("RGB")  # Remove alpha channel

        if self.transforms:
            image = self.transforms(image)

        label = row["label"]

        return image, label


class CellClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes,
        learning_rate=1e-3,
        model_name="resnet50",
        use_pretrained=True,
        class_names=None,
        weight_decay=1e-4,  # noqa: ARG002
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.class_weights = class_weights
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.use_pretrained = use_pretrained
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # Load model with or without pretrained weights
        if model_name == "resnet50":
            if use_pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_name == "efficientnet_b0":
            if use_pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        elif model_name == "vit_b_16":
            if use_pretrained:
                self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            else:
                self.backbone = models.vit_b_16(weights=None)
            self.backbone.heads.head = nn.Linear(self.backbone.heads.head.in_features, num_classes)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Store predictions for confusion matrix
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        logits = self.forward(x)

        # Use weighted loss if class weights provided
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        else:
            loss = F.cross_entropy(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        f1 = self.train_f1(preds, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        logits = self.forward(x)

        # Use weighted loss for validation too
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        else:
            loss = F.cross_entropy(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        f1 = self.test_f1(preds, y)

        # Store predictions for confusion matrix
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """Create and log confusion matrix after test epoch"""
        if len(self.test_predictions) > 0:
            # Create confusion matrix
            cm = confusion_matrix(self.test_targets, self.test_predictions)

            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names
            )
            plt.title("Confusion Matrix - Test Set")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Log to wandb
            if self.logger and hasattr(self.logger, "experiment"):
                # Log confusion matrix
                self.logger.experiment.log(
                    {
                        "confusion_matrix": wandb.Image(plt),
                        "confusion_matrix_data": wandb.Table(
                            data=cm.tolist(), columns=self.class_names, rows=self.class_names
                        ),
                    }
                )

                # Log final computed metrics
                test_acc_final = self.test_acc.compute().item()
                test_f1_final = self.test_f1.compute().item()

                self.logger.experiment.log({"test_acc_final": test_acc_final, "test_f1_final": test_f1_final})

                # Also log summary metrics
                self.logger.experiment.summary.update({"test_acc": test_acc_final, "test_f1": test_f1_final})

            plt.close()

            # Clear stored predictions
            self.test_predictions = []
            self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def log_example_images(logger, data_module, n_per_class=2):
    """Log example images from each class to wandb"""
    if not logger or not hasattr(logger, "experiment"):
        return

    example_images = data_module.get_example_images(n_per_class)

    # Create a single organized view with all classes
    all_images = []
    for img_data in example_images:
        all_images.append(wandb.Image(img_data["image"], caption=f"{img_data['class_name']}"))

    # Log all images in a single table/gallery
    logger.experiment.log({"class_examples": all_images})


def export_to_onnx(model, data_module, model_name, img_size):
    """Export the trained model to ONNX format"""

    # Set model to evaluation mode and move to CPU
    model.eval()
    model = model.cpu()

    # Create dummy input tensor on CPU
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Create output directory
    onnx_dir = f"onnx_models/treatment_concentration_{model_name}"
    os.makedirs(onnx_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Save class mapping and metadata
    metadata = {
        "model_name": model_name,
        "num_classes": data_module.num_classes,
        "class_names": [str(name) for name in data_module.class_names],
        "class_mapping": {str(i): str(class_name) for i, class_name in enumerate(data_module.class_names)},
        "img_size": img_size,
        "input_shape": [1, 3, img_size, img_size],
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }

    metadata_path = os.path.join(onnx_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel exported to ONNX format:")
    print(f"   Model: {onnx_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"   Input shape: {metadata['input_shape']}")
    print(f"   Classes: {metadata['num_classes']}")

    return onnx_path, metadata_path


def train_model(
    csv_path,
    model_name="resnet50",
    img_size=224,
    batch_size=32,
    max_epochs=50,
    gpus=1,
    use_pretrained=True,
    learning_rate=1e-3,
    weight_decay=1e-4,
    loss_patience=7,
    f1_patience=5,
    min_delta=0.001,
    project_name="cell-classification",
    run_name=None,
):
    """Train a cell classification model using PyTorch Lightning"""

    start_time = time.time()  # Track total training time

    # Create data module
    data_module = CellDataModule(csv_path=csv_path, img_size=img_size, batch_size=batch_size)

    # Setup data (this creates the train/val/test splits)
    data_module.setup()

    # Create model
    model = CellClassifier(
        num_classes=data_module.num_classes,
        learning_rate=learning_rate,
        model_name=model_name,
        use_pretrained=use_pretrained,
        class_names=data_module.class_names,
        weight_decay=weight_decay,
        class_weights=data_module.class_weights,
    )

    print("\nModel configuration:")
    print(f"  Architecture: {model_name}")
    print(f"  Pretrained weights: {'Yes' if use_pretrained else 'No'}")
    print(f"  Number of classes: {data_module.num_classes}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Loss patience: {loss_patience}")
    print(f"  F1 patience: {f1_patience}")
    print(f"  Min delta: {min_delta}")

    # Create run name if not provided
    if run_name is None:
        run_name = f"{model_name}_{'pretrained' if use_pretrained else 'scratch'}_wd{weight_decay}"

    # Wandb Logger
    logger = WandbLogger(project=project_name, name=run_name, save_dir="logs")

    # Log example images
    print("Logging example images to wandb...")
    log_example_images(logger, data_module, n_per_class=2)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/treatment_concentration_{model_name}{'_pretrained' if use_pretrained else '_scratch'}",
        filename="{epoch}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
    )

    # Early stopping callbacks with F1 monitoring
    early_stopping_loss = EarlyStopping(
        monitor="val_loss", patience=loss_patience, mode="min", verbose=True, check_on_train_epoch_end=False
    )

    early_stopping_f1 = EarlyStopping(
        monitor="val_f1",
        patience=f1_patience,
        mode="max",
        min_delta=min_delta,
        verbose=True,
        check_on_train_epoch_end=False,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stopping_loss, early_stopping_f1],
        logger=logger,
        deterministic=True,
    )

    # Train model
    trainer.fit(model, data_module)

    print("Optimizer state:")
    for param_group in model.optimizers().optimizer.param_groups:
        print(f"Weight decay: {param_group['weight_decay']}")

    # Test model
    test_results = trainer.test(model, data_module)

    # Log final test results to wandb explicitly
    if logger and hasattr(logger, "experiment"):
        # Extract test results from the returned metrics
        test_metrics = test_results[0] if test_results else {}

        # Log all test metrics explicitly
        metrics = {
            "test_loss": test_metrics.get("test_loss", 0.0),
            "test_acc": test_metrics.get("test_acc", 0.0),
            "test_f1": test_metrics.get("test_f1", 0.0),
            "final_epoch": trainer.current_epoch,
            "total_training_time": time.time() - start_time,
            "best_model_path": checkpoint_callback.best_model_path,
            "training_completed": True,
            "weight_decay": weight_decay,
        }

        logger.experiment.log(metrics)

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS:")
        print("=" * 50)
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
        print(f"  Test F1 Score: {metrics['test_f1']:.4f}")
        print(f"  Final Epoch: {metrics['final_epoch']}")
        print(f"  Best Model: {metrics['best_model_path']}")
        print("=" * 50)

    # Print class mapping
    print("\nClass mapping:")
    for i, class_name in enumerate(data_module.class_names):
        print(f"{i}: {class_name}")

    # Load best model for ONNX export
    best_model = CellClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_classes=data_module.num_classes,
        model_name=model_name,
        use_pretrained=use_pretrained,
        class_names=data_module.class_names,
        weight_decay=weight_decay,
        class_weights=data_module.class_weights,
    )

    # Export to ONNX
    onnx_path, metadata_path = export_to_onnx(best_model, data_module, model_name, img_size)

    # Log ONNX export info to wandb
    if logger and hasattr(logger, "experiment"):
        logger.experiment.log({"onnx_model_path": onnx_path, "onnx_metadata_path": metadata_path})

    # Finish wandb run
    wandb.finish()

    return model, data_module, trainer, onnx_path


def main():
    parser = argparse.ArgumentParser(description="Cell Classification with PyTorch Lightning and Wandb")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_b_16"],
        help="Model architecture to use",
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch without pretrained weights")

    # Early stopping parameters
    parser.add_argument("--loss_patience", type=int, default=7, help="Patience for validation loss early stopping")
    parser.add_argument("--f1_patience", type=int, default=5, help="Patience for validation F1 early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change to qualify as improvement")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer (default: 1e-3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for L2 regularization (default: 1e-4)"
    )

    # Wandb parameters
    parser.add_argument("--project_name", type=str, default="cell-classification", help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()

    # Train model
    model, data_module, trainer, onnx_path = train_model(
        csv_path=args.csv_path,
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        use_pretrained=not args.no_pretrained,
        weight_decay=args.weight_decay,
        loss_patience=args.loss_patience,
        f1_patience=args.f1_patience,
        min_delta=args.min_delta,
        project_name=args.project_name,
        run_name=args.run_name,
    )

    print(f"\nTraining completed! ONNX model saved at: {onnx_path}")


if __name__ == "__main__":
    main()
