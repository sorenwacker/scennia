import os
import re
import traceback
from dataclasses import dataclass, field
from os.path import exists, isfile, join

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile
from pydantic import BaseModel, Field
from pydantic_core import to_json


# Get the stem of given file name
def file_stem(file_name: str) -> str:
    stem, _ = os.path.splitext(file_name)
    return stem


# Get the extension, excluding the '.', of given file name. Returns empty string if file has no extension.
def file_extension(file_name: str) -> str:
    _, ext = os.path.splitext(file_name)
    if len(ext) > 0:
        ext = ext[1:]
    return ext


# Parse the actual lactate concentration from given file stem.
def parse_actual_lactate_concentration(file_stem: str) -> int | None:
    matches = re.search(r"[\S]+ [\S]+ ([^_])_[^_]+_[^_]+", file_stem)
    if matches is not None:
        char = matches.group(1)
        match char.upper():
            case "A":
                return 0
            case "B":
                return 0
            case "C":
                return 5
            case "D":
                return 10
            case "E":
                return 20
            case "F":
                return 40
            case "G":
                return 40
            case "H":
                return 80
    return None


# Meta-data for an image
class ImageMetaData(BaseModel):
    # File name of the image
    file_name: str
    # Stem of the file name
    file_stem: str = Field(default_factory=lambda data: os.path.splitext(data["file_name"])[0])
    # Extension, excluding the '.', of the file. Empty string if file has no extension.
    file_extension: str = Field(default_factory=lambda data: file_extension(data["file_name"]))
    # Actual lacate concentration of the image, in millimolar (mM)
    actual_lactate_concentration: int | None = Field(
        default_factory=lambda data: parse_actual_lactate_concentration(data["file_stem"])
    )


# Compressed and img-src (base64) encoded image
class EncodedImage(BaseModel):
    # Encoded image string
    contents: str
    # Image width
    width: int
    # Image height
    height: int


# Data for an image
class ImageData(BaseModel):
    # Image metadata
    meta_data: ImageMetaData | None = None
    # Compressed and img-src encoded image
    encoded_image: EncodedImage

    def actual_lactate_concentration(self) -> int | None:
        if self.meta_data is None:
            return None
        return self.meta_data.actual_lactate_concentration


# Predicted cell data
class CellPrediction(BaseModel):
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    treatment_type: str
    concentration: int
    all_predictions: list[float]


# Cell data
class Cell(BaseModel):
    id: int
    centroid_y: float
    centroid_x: float
    area: float
    perimeter: float
    eccentricity: float
    bbox: list[int]
    is_large: bool
    contour: list[list[float]]
    predicted_properties: CellPrediction | None = None

    # Gets the predicted and relative lactate concentration of the cell, along with the confidence of the prediction.
    def lactate_concentration(self, actual_concentration: int | None) -> tuple[int | None, int | None, float | None]:
        concentration = None
        confidence = None
        if self.predicted_properties:
            concentration = self.predicted_properties.concentration
            confidence = self.predicted_properties.confidence
        r_concentration = None
        if concentration is not None and actual_concentration is not None:
            r_concentration = concentration - actual_concentration
        return (concentration, r_concentration, confidence)


# Turns a relative lactate concentration into an English conclusion whether the cell is lactate resistance or not.
def relative_lactate_concentration_into_resistance(r_concentration: int) -> str:
    if r_concentration <= -40:
        return "very likely lactate resistant"
    if r_concentration < 0:
        return "likely lactate resistant"
    return "NOT lactate resistant"


# Processed data for an image with cells
class ProcessedData(BaseModel):
    # Cropped compressed and img-src encoded images, per cell ID.
    cropped_encoded_images: dict[int, EncodedImage]
    # Cells detected in the image, per cell ID.
    cells: dict[int, Cell]
    # Aggregate data about the image
    # Median cell area
    median_area: float = Field(default_factory=lambda data: float(np.median([c.area for c in data["cells"].values()])))
    # Mean cell area
    mean_area: float = Field(default_factory=lambda data: float(np.mean([c.area for c in data["cells"].values()])))


# Prepared image
@dataclass
class PreparedImage:
    # Compressed image
    compressed_image: ImageFile
    # Hash of the original image
    hash: str


# Simple file-based cache system
class DiskCache:
    def __init__(self):
        self.cache_dir = "cache"

    def __cache_path(self, *directories: str) -> str:
        path = self.cache_dir
        for directory in directories:
            path = join(path, directory)
        return path

    def __create_cache_path(self, file_name: str, *directories: str) -> str:
        path = self.__cache_path(*directories)
        os.makedirs(path, exist_ok=True)
        return join(path, file_name)

    def __cache_path_exists(self, file_name: str, *directories: str) -> str | None:
        path = self.__cache_path(*directories)
        path = join(path, file_name)
        if exists(path):
            return path
        return None

    # Load image data
    def load_image_data(self, hash: str) -> ImageData | None:
        try:
            path = self.__cache_path_exists("image_data.json", hash)
            if path is not None:
                with open(path) as f:
                    return ImageData.model_validate_json(f.read())
        except Exception as e:
            msg = "Error loading image data from disk cache"
            raise Exception(msg) from e
        return None

    # Save image data
    def save_image_data(self, hash: str, data: ImageData):
        json = to_json(data, indent=2)
        try:
            path = self.__create_cache_path("image_data.json", hash)
            with open(path, "wb") as f:
                f.write(json)
        except Exception as e:
            msg = "Error saving image data to disk cache"
            raise Exception(msg) from e

    # Load processed data
    def load_processed_data(self, hash: str) -> ProcessedData | None:
        try:
            path = self.__cache_path_exists("processed_data.json", hash)
            if path is not None:
                with open(path) as f:
                    return ProcessedData.model_validate_json(f.read())
        except Exception as e:
            msg = "Error loading processed data from disk cache"
            raise Exception(msg) from e
        return None

    # Save processed data
    def save_processed_data(self, hash: str, data: ProcessedData):
        json = to_json(data, indent=2)
        try:
            path = self.__create_cache_path("processed_data.json", hash)
            with open(path, "wb") as f:
                f.write(json)
        except Exception as e:
            msg = "Error saving processed data to disk cache"
            raise Exception(msg) from e

    # Load uncompressed image from disk cache
    def load_uncompressed_image(self, hash: str, extension: str) -> ImageFile | None:
        try:
            path = self.__cache_path_exists(f"uncompressed.{extension}", hash)
            if path is not None:
                return Image.open(path)
        except Exception as e:
            msg = "Error loading uncompressed image from disk cache"
            raise RuntimeError(msg) from e
        return None

    # Save uncompressed image to disk cache
    def save_uncompressed_image(self, hash: str, extension: str, image: Image.Image):
        try:
            path = self.__create_cache_path(f"uncompressed.{extension}", hash)
            image.save(path)
        except Exception as e:
            msg = "Error saving uncompressed image to disk cache"
            raise Exception(msg) from e

    # Load compressed image from disk cache
    def load_compressed_image(self, hash: str) -> ImageFile | None:
        try:
            path = self.__cache_path_exists("compressed.webp", hash)
            if path is not None:
                return Image.open(path)
        except Exception as e:
            msg = "Error loading compressed image from disk cache"
            raise RuntimeError(msg) from e
        return None

    # Save compressed image to disk cache
    def save_compressed_image(self, hash: str, image: Image.Image):
        try:
            path = self.__create_cache_path("compressed.webp", hash)
            image.save(path)
        except Exception as e:
            msg = "Error saving compressed image to disk cache"
            raise Exception(msg) from e

    # Load compressed cropped image from disk cache
    def load_cropped_compressed_image(self, hash: str, cell_id: str) -> ImageFile | None:
        try:
            path = self.__cache_path_exists(f"{cell_id}.webp", hash, "cropped")
            if path is not None:
                return Image.open(path)
        except Exception as e:
            msg = "Error loading cropped compressed image from disk cache"
            raise Exception(msg) from e
        return None

    # Save compressed cropped images to disk cache
    def save_cropped_compressed_images(self, hash: str, images: dict[int, Image.Image]):
        try:
            for id, image in images.items():
                path = self.__create_cache_path(f"{id}.webp", hash, "cropped")
                image.save(path)
        except Exception as e:
            msg = "Error saving cropped compressed images to disk cache"
            raise Exception(msg) from e

    # Load compressed image from disk cache
    def load_compressed_images(self) -> list[PreparedImage]:
        images = []
        for p in os.listdir(self.cache_dir):
            path = join(self.cache_dir, p, "compressed.webp")
            try:
                if isfile(path):
                    image = Image.open(path)
                    images.append(PreparedImage(compressed_image=image, hash=p))
            except Exception as e:
                msg = "Error loading compressed images from disk cache"
                raise RuntimeError(msg) from e
        return images


@dataclass
class DataManager:
    disk_cache: DiskCache = field(default_factory=DiskCache)

    uncompressed_images: dict[str, ImageFile] = field(default_factory=dict)
    compressed_images: dict[str, ImageFile] = field(default_factory=dict)
    cropped_compressed_images: dict[str, dict[int, Image.Image]] = field(default_factory=dict)

    image_data: dict[str, ImageData] = field(default_factory=dict)
    processed_data: dict[str, ProcessedData] = field(default_factory=dict)

    prepared_images: list[PreparedImage] | None = None

    def __try_load_uncompressed_image(self, hash: str, extension: str):
        if hash not in self.uncompressed_images:
            try:
                print(f"Loading uncompressed image for hash '{hash}' from disk")
                uncompressed_image = self.disk_cache.load_uncompressed_image(hash, extension)
                if uncompressed_image is not None:
                    self.uncompressed_images[hash] = uncompressed_image
            except Exception:
                print(f"Failed to load uncompressed image for hash '{hash}'; continuing without image loaded")
                traceback.print_exc()

    # Get uncompressed image by hash and extension
    def get_uncompressed_image(self, hash: str, extension: str) -> ImageFile | None:
        self.__try_load_uncompressed_image(hash, extension)
        return self.uncompressed_images.get(hash)

    # Set uncompressed image by hash
    def set_uncompressed_image(self, hash: str, image: ImageFile):
        self.uncompressed_images[hash] = image

    # Set and save uncompressed image by hash and extension
    def save_uncompressed_image(self, hash: str, extension: str, image: ImageFile):
        self.set_uncompressed_image(hash, image)
        try:
            print(f"Saving uncompressed image for hash '{hash}' to disk")
            self.disk_cache.save_uncompressed_image(hash, extension, image)
        except Exception:
            print(f"Failed to save uncompressed image for hash '{hash}'; continuing without image saved")
            traceback.print_exc()

    def __try_load_compressed_image(self, hash: str):
        if hash not in self.compressed_images:
            try:
                print(f"Loading compressed image for hash '{hash}' from disk")
                compressed_image = self.disk_cache.load_compressed_image(hash)
                if compressed_image is not None:
                    self.compressed_images[hash] = compressed_image
            except Exception:
                print(f"Failed to load compressed image for hash '{hash}'; continuing without image loaded")
                traceback.print_exc()

    # Get compressed image by hash
    def get_compressed_image(self, hash: str) -> ImageFile | None:
        self.__try_load_compressed_image(hash)
        return self.compressed_images.get(hash)

    # Set compressed image by hash
    def set_compressed_image(self, hash: str, image: ImageFile):
        self.compressed_images[hash] = image

    # Set and save compressed image by hash
    def save_compressed_image(self, hash: str, image: ImageFile):
        self.compressed_images[hash] = image
        try:
            print(f"Saving compressed image for hash '{hash}' to disk")
            self.disk_cache.save_compressed_image(hash, image)
        except Exception:
            print(f"Failed to save compressed image for hash '{hash}'; continuing without image saved")
            traceback.print_exc()

    # Get compressed image by hash
    def get_cropped_compressed_images(self, hash: str) -> dict[int, Image.Image] | None:
        return self.cropped_compressed_images.get(hash)

    # Set compressed image by hash
    def set_cropped_compressed_images(self, hash: str, images: dict[int, Image.Image]):
        self.cropped_compressed_images[hash] = images

    # Set and save compressed image by hash
    def save_cropped_compressed_images(self, hash: str, images: dict[int, Image.Image]):
        self.cropped_compressed_images[hash] = images
        try:
            print(f"Saving cropped compressed images for hash '{hash}' to disk")
            self.disk_cache.save_cropped_compressed_images(hash, images)
        except Exception:
            print(f"Failed to save cropped compressed images for hash '{hash}'; continuing without images saved")
            traceback.print_exc()

    def __try_load_image_data(self, hash: str):
        if hash not in self.image_data:
            try:
                print(f"Loading image data for hash '{hash}' from disk")
                image_data = self.disk_cache.load_image_data(hash)
                if image_data is not None:
                    self.image_data[hash] = image_data
            except Exception:
                print(f"Failed to load image data for hash '{hash}'; continuing without data loaded")
                traceback.print_exc()

    # Get image data by hash
    def get_image_data(self, hash: str) -> ImageData | None:
        self.__try_load_image_data(hash)
        return self.image_data.get(hash)

    # Update image data by hash, and get the updated image data
    def update_image_data(self, hash: str, image_data: ImageData) -> ImageData:
        self.__try_load_image_data(hash)
        cached_image_data = self.image_data.get(hash)
        if cached_image_data is None:
            self.image_data[hash] = image_data
            return image_data
        if image_data.meta_data is not None:
            cached_image_data.meta_data = image_data.meta_data
        cached_image_data.encoded_image = image_data.encoded_image
        return cached_image_data

    # Update and save image data, and get the updated image data
    def save_image_data(self, hash: str, image_data: ImageData) -> ImageData:
        updated_image_data = self.update_image_data(hash, image_data)
        try:
            print(f"Saving image data for hash '{hash}' to disk")
            self.disk_cache.save_image_data(hash, image_data)
        except Exception:
            print(f"Failed to save image data for hash '{hash}'; continuing without data saved")
            traceback.print_exc()
        return updated_image_data

    def __try_load_processed_data(self, hash: str):
        if hash not in self.processed_data:
            try:
                print(f"Loading processed data for hash '{hash}' from disk")
                processed_data = self.disk_cache.load_processed_data(hash)
                if processed_data is not None:
                    self.processed_data[hash] = processed_data
            except Exception:
                print(f"Failed to load processed data for hash '{hash}'; continuing without data loaded")
                traceback.print_exc()

    # Get processed data by hash
    def get_processed_data(self, hash: str) -> ProcessedData | None:
        self.__try_load_processed_data(hash)
        return self.processed_data.get(hash)

    # Update processed data by hash
    def update_processed_data(self, hash: str, processed_data: ProcessedData) -> ProcessedData:
        self.__try_load_processed_data(hash)
        cached_processed_data = self.processed_data.get(hash)
        if cached_processed_data is None:
            self.processed_data[hash] = processed_data
            return processed_data
        cached_processed_data.cropped_encoded_images = processed_data.cropped_encoded_images
        cached_processed_data.cells = processed_data.cells
        cached_processed_data.median_area = processed_data.median_area
        cached_processed_data.mean_area = processed_data.mean_area
        return cached_processed_data

    # Update and save processed data by hash
    def save_processed_data(self, hash: str, processed_data: ProcessedData) -> ProcessedData:
        updated_processed_data = self.update_processed_data(hash, processed_data)
        try:
            print(f"Saving processed data for hash '{hash}' to disk")
            self.disk_cache.save_processed_data(hash, processed_data)
        except Exception:
            print(f"Failed to save processed data for hash '{hash}'; continuing without data saved")
            traceback.print_exc()
        return updated_processed_data

    # Get prepared images
    def get_prepared_images(self, reload: bool = False) -> list[PreparedImage]:
        if reload or self.prepared_images is None:
            self.prepared_images = []
            try:
                self.prepared_images = self.disk_cache.load_compressed_images()
            except Exception:
                print("Failed to load prepared images; continuing without data loaded")
                traceback.print_exc()
            # Cache compressed images
            for prepared_image in self.prepared_images:
                self.set_compressed_image(prepared_image.hash, prepared_image.compressed_image)
        return self.prepared_images

    # Get prepared image at index
    def get_prepared_image(self, index: int) -> PreparedImage | None:
        prepared_images = self.get_prepared_images()
        try:
            return prepared_images[index]
        except IndexError:
            return None
