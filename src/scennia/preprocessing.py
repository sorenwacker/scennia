import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from cellpose import models
from PIL import Image
from skimage.measure import regionprops
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Cell Image Preprocessing Pipeline")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the treatment folders with images"
    )
    parser.add_argument(
        "--output_dir", type=str, default="processed_dataset", help="Directory to save processed cells and metadata"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images per folder (for testing)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for cellpose model")

    args = parser.parse_args()

    print(f"Starting preprocessing of images from: {args.data_dir}")

    # Initialize the preprocessor
    preprocessor = CellPreprocessor(base_dir=args.data_dir, output_dir=args.output_dir, use_gpu=args.gpu)

    # Process all folders
    _ = preprocessor.process_all_folders(limit_per_folder=args.limit)

    # Create consolidated dataset
    df = preprocessor.create_training_dataset()

    print(f"Preprocessing complete! Dataset available at: {os.path.join(args.output_dir, 'dataset.csv')}")
    print(f"Total cells extracted: {len(df)}")

    # Print distribution of cells across treatment conditions
    print("\nCell distribution by treatment:")
    for treatment, group in df.groupby(["treatment_type", "concentration"]):
        treatment_type, concentration = treatment
        print(f"  - {treatment_type} (concentration: {concentration}): {len(group)} cells")


class CellPreprocessor:
    def __init__(self, base_dir, output_dir="processed_dataset", use_gpu=True):
        """
        Initialize the cell preprocessing pipeline

        Args:
            base_dir (str): Base directory containing folders with images
            output_dir (str): Directory to save processed cells and metadata
            use_gpu (bool): Whether to use GPU for cellpose model
        """
        self.base_dir = base_dir
        self.output_dir = output_dir

        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Initialize cellpose model
        print("Initializing cellpose model (GPU:" + ("enabled" if use_gpu else "disabled") + ")")
        self.model = models.CellposeModel(gpu=use_gpu)

        # Treatment info extraction (from folder names)
        self.treatment_patterns = {
            "control": re.compile(r"Control (\d+)k-cm2"),
            "lac": re.compile(r"Lac(\d+) (\d+)k-cm2"),
            "dox": re.compile(r"Dox(\d+\.?\d*) (\d+\.?\d*)k-cm2"),
        }

        # Store all cell data for final CSV
        self.all_cells = []

    def extract_treatment_info(self, folder_name):
        """Extract treatment type, concentration, and density from folder name"""
        treatment_type = None
        concentration = 0
        density = 0

        # Check for control
        control_match = self.treatment_patterns["control"].search(folder_name)
        if control_match:
            treatment_type = "control"
            concentration = 0
            density = float(control_match.group(1))
            return treatment_type, concentration, density

        # Check for lactate
        lac_match = self.treatment_patterns["lac"].search(folder_name)
        if lac_match:
            treatment_type = "lactate"
            concentration = float(lac_match.group(1))
            density = float(lac_match.group(2))
            return treatment_type, concentration, density

        # Check for doxorubicin
        dox_match = self.treatment_patterns["dox"].search(folder_name)
        if dox_match:
            treatment_type = "doxorubicin"
            concentration = float(dox_match.group(1))
            density = float(dox_match.group(2))
            return treatment_type, concentration, density

        # If no match found, try to extract any information
        print(f"Warning: Could not fully parse folder name: {folder_name}")
        if "Control" in folder_name:
            treatment_type = "control"
        elif "Lac" in folder_name:
            treatment_type = "lactate"
            # Try to extract number after Lac
            lac_num = re.search(r"Lac(\d+)", folder_name)
            if lac_num:
                concentration = float(lac_num.group(1))
        elif "Dox" in folder_name:
            treatment_type = "doxorubicin"
            # Try to extract number after Dox
            dox_num = re.search(r"Dox(\d+\.?\d*)", folder_name)
            if dox_num:
                concentration = float(dox_num.group(1))

        # Try to extract density
        density_match = re.search(r"(\d+\.?\d*)k-cm2", folder_name)
        if density_match:
            density = float(density_match.group(1))

        return treatment_type, concentration, density

    def process_folder(self, folder_path, limit=None):
        """
        Process all images in a folder

        Args:
            folder_path (str): Path to folder containing images
            limit (int, optional): Limit number of images to process (for testing)

        Returns:
            (dict): Summary of processing results
        """
        folder_name = os.path.basename(folder_path)
        print(f"Processing folder: {folder_name}")

        # Extract treatment information
        treatment_type, concentration, density = self.extract_treatment_info(folder_name)

        if not treatment_type:
            print(f"Could not extract treatment info from folder: {folder_name}")
            return None

        # Create a clean folder name for output
        clean_folder_name = folder_name.replace(" ", "_").replace("-", "_")

        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if limit:
            image_files = image_files[:limit]

        if not image_files:
            print(f"No image files found in {folder_path}")
            return None

        print(f"Found {len(image_files)} images to process")

        # Process each image
        results = []
        cell_count = 0

        for img_path in tqdm(image_files, desc=f"Processing {folder_name}"):
            img_result = self.process_image(
                img_path, clean_folder_name, treatment_type=treatment_type, concentration=concentration, density=density
            )

            if img_result:
                results.append(img_result)
                cell_count += img_result["cell_count"]

        # Create summary
        summary = {
            "folder": folder_name,
            "treatment_type": treatment_type,
            "concentration": concentration,
            "density": density,
            "image_count": len(image_files),
            "processed_image_count": len(results),
            "cell_count": cell_count,
            "results": results,
        }

        # Save summary to JSON
        summary_path = os.path.join(self.metadata_dir, f"{clean_folder_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Extracted {cell_count} cells from {len(results)} images in {folder_name}")

        return summary

    def process_image(self, img_path, folder_prefix, treatment_type, concentration, density):
        """
        Process a single image, segment cells, and save cropped cells

        Args:
            img_path (str): Path to image file
            folder_prefix (str): Prefix for output folder structure
            treatment_type (str): Type of treatment
            concentration (float): Concentration of treatment
            density (float): Cell density

        Returns:
            (dict): Processing results for this image
        """
        try:
            # Load the image
            img = np.array(Image.open(img_path))
            os.path.basename(img_path)

            # Use a hash of the image data for unique identification
            img_hash = hashlib.md5(img.tobytes()).hexdigest()

            # Run cellpose segmentation
            flow_threshold = 0.4
            cell_prob_threshold = 0.0

            result = self.model.eval([img], flow_threshold=flow_threshold, cellprob_threshold=cell_prob_threshold)

            mask = result[0][0]

            if mask.max() == 0:
                print(f"No cells detected in {img_path}")
                return None

            # Get cell properties
            props = regionprops(mask)

            # Calculate metadata
            median_area = np.median([p.area for p in props]) if props else 0

            # Process and save each cell
            cell_metadata = []

            for i, prop in enumerate(props):
                cell_id = i + 1
                is_large = prop.area > median_area

                # Extract bounding box with padding
                y0, x0, y1, x1 = prop.bbox
                padding = 10
                y0 = max(0, y0 - padding)
                x0 = max(0, x0 - padding)
                y1 = min(img.shape[0], y1 + padding)
                x1 = min(img.shape[1], x1 + padding)

                # Create cropped cell image
                cell_img = img[y0:y1, x0:x1]

                # Create unique filename for this cell
                cell_filename = f"{img_hash}_{cell_id:03d}.png"
                cell_path = os.path.join(self.images_dir, cell_filename)

                # Save the cell image
                Image.fromarray(cell_img).save(cell_path)

                # Create cell metadata for the training dataset
                cell_data = {
                    "cell_path": cell_path,
                    "cell_id": cell_id,
                    "source_image": img_path,
                    "source_folder": folder_prefix,
                    "treatment_type": treatment_type,
                    "concentration": concentration,
                    "density": density,
                    "area": float(prop.area),
                    "perimeter": float(prop.perimeter),
                    "eccentricity": float(prop.eccentricity),
                    "centroid_y": float(prop.centroid[0]),
                    "centroid_x": float(prop.centroid[1]),
                    "is_large": bool(is_large),
                    "image_hash": img_hash,
                }

                # Add to the overall dataset
                self.all_cells.append(cell_data)
                cell_metadata.append(cell_data)

            # Save metadata for this image
            metadata_path = os.path.join(self.metadata_dir, f"{folder_prefix}_{img_hash[:8]}_metadata.json")
            metadata = {
                "source_image": img_path,
                "image_hash": img_hash,
                "treatment_type": treatment_type,
                "concentration": concentration,
                "density": density,
                "median_area": float(median_area),
                "cell_count": len(props),
                "cells": cell_metadata,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Return processing results
            return {
                "source_image": img_path,
                "image_hash": img_hash,
                "cell_count": len(props),
                "metadata_path": metadata_path,
            }

        except Exception as e:
            print(f"Error processing {img_path}: {e!s}")
            return None

    def create_training_dataset(self, output_file="dataset.csv"):
        """
        Create a consolidated CSV file for training

        Args:
            output_file (str): Output CSV file path

        Returns:
            (pd.DataFrame): Consolidated dataset
        """
        print("Creating training dataset CSV...")

        if not self.all_cells:
            print("No cells found! Make sure to run process_all_folders first.")
            return pd.DataFrame()

        # Create dataframe
        df = pd.DataFrame(self.all_cells)

        # Save to CSV
        csv_path = os.path.join(self.output_dir, output_file)
        df.to_csv(csv_path, index=False)

        print(f"Training dataset created with {len(df)} cells, saved to {csv_path}")

        # Generate statistics
        print("\nDataset Statistics:")
        print(f"Total cells: {len(df)}")
        print(f"Treatment types: {df['treatment_type'].unique()}")

        print("\nCells per treatment type:")
        treatment_counts = df.groupby("treatment_type").size()
        for treatment, count in treatment_counts.items():
            print(f"  {treatment}: {count}")

        print("\nCells per treatment concentration:")
        concentration_counts = df.groupby(["treatment_type", "concentration"]).size()
        for (treatment, conc), count in concentration_counts.items():
            print(f"  {treatment} {conc}: {count}")

        print("\nCells per density:")
        density_counts = df.groupby("density").size()
        for density, count in density_counts.items():
            print(f"  {density}k-cm2: {count}")

        # Verify all image paths exist
        missing_files = []
        for _idx, row in df.iterrows():
            if not os.path.exists(row["cell_path"]):
                missing_files.append(row["cell_path"])

        if missing_files:
            print(f"\nWarning: {len(missing_files)} image files are missing!")
            print("First few missing files:")
            for f in missing_files[:5]:
                print(f"  {f}")
        else:
            print("\n✅ All cell image paths verified!")

        return df

    def process_all_folders(self, limit_per_folder=None):
        """
        Process all folders in the base directory

        Args:
            limit_per_folder (int, optional): Limit images per folder (for testing)

        Returns:
            (list): Summary of processing results
        """
        # Get all folders
        folders = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                folders.append(item)

        if not folders:
            print(f"No folders found in {self.base_dir}")
            return []

        print(f"Found {len(folders)} folders to process:")
        for folder in folders:
            treatment_type, concentration, density = self.extract_treatment_info(folder)
            print(f"  - {folder}: {treatment_type} (conc: {concentration}, density: {density})")

        results = []
        for folder in folders:
            folder_path = os.path.join(self.base_dir, folder)
            result = self.process_folder(folder_path, limit=limit_per_folder)
            if result:
                results.append(result)

        return results


if __name__ == "__main__":
    main()
