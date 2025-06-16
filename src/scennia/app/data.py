import os
from dataclasses import dataclass, field
from os.path import exists, isfile, join, splitext

from PIL import Image
from PIL.ImageFile import ImageFile
from pydantic import BaseModel, Field
from pydantic_core import from_json, to_json


class EncodedImage(BaseModel):
    contents: str
    width: int
    height: int


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
    predicted_properties: dict = Field(default_factory=dict)


class AggregateData(BaseModel):
    median_area: float = 0.0


@dataclass
class ImageData:
    uncompressed_image: ImageFile
    encoded_image: EncodedImage
    cropped_encoded_images: dict[str, EncodedImage] = field(default_factory=dict)
    cells: list[Cell] = field(default_factory=list)
    aggregate_data: AggregateData = field(default_factory=AggregateData)


class DiskCacheData(BaseModel):
    cells: list[Cell] = field(default_factory=list)
    aggregate_data: AggregateData = field(default_factory=AggregateData)


@dataclass
class ImageWithHash:
    image: ImageFile
    hash: str


# Simple file-based cache system
class DiskCache:
    def __init__(self):
        self.CACHE_DIR = "cache"
        self.CROPPED_CACHE_DIR = join(self.CACHE_DIR, "cropped")
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.CROPPED_CACHE_DIR, exist_ok=True)

    # Load result from disk cache
    def load_data(self, image_hash: str) -> DiskCacheData | None:
        cache_file = join(self.CACHE_DIR, f"{image_hash}.json")
        try:
            if exists(cache_file):
                with open(cache_file) as f:
                    return DiskCacheData.model_validate(from_json(f.read()))
        except Exception as e:
            print(f"Error loading data from disk cache: {e!s}")
        return None

    # Save results to disk cache
    def save_data(self, image_hash: str, cells: list[Cell], aggregate_data: AggregateData):
        data = DiskCacheData(cells=cells, aggregate_data=aggregate_data)
        json = to_json(data, indent=2)
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            cache_file = join(self.CACHE_DIR, f"{image_hash}.json")
            with open(cache_file, "wb") as f:
                f.write(json)
            return True
        except Exception as e:
            print(f"Error saving data to disk cache: {e!s}")
            return False

    # Load compressed image from disk cache
    def load_compressed_image(self, image_hash: str) -> ImageFile | None:
        crop_file = join(self.CACHE_DIR, f"{image_hash}.webp")
        try:
            if exists(crop_file):
                return Image.open(crop_file)
        except Exception as e:
            print(f"Error loading compressed cropped image from disk cache: {e!s}")
        return None

    # Save compressed image to disk cache
    def save_compressed_image(self, image_hash: str, image: Image.Image):
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = join(self.CACHE_DIR, f"{image_hash}.webp")
            image.save(path)
            return True
        except Exception as e:
            print(f"Error saving compressed cropped image to disk cache: {e!s}")
            return False

    # Load compressed cropped image from disk cache
    def load_compressed_cropped_image(self, image_hash: str, cell_id: str) -> ImageFile | None:
        crop_file = join(self.CROPPED_CACHE_DIR, f"{image_hash}_{cell_id}.webp")
        try:
            if exists(crop_file):
                return Image.open(crop_file)
        except Exception as e:
            print(f"Error loading compressed cropped image from disk cache: {e!s}")
        return None

    # Save compressed cropped images to disk cache
    def save_compressed_cropped_images(self, image_hash: str, cropped_uncompressed_images: dict[str, Image.Image]):
        try:
            os.makedirs(self.CROPPED_CACHE_DIR, exist_ok=True)
            for cell_id, image in cropped_uncompressed_images.items():
                path = join(self.CROPPED_CACHE_DIR, f"{image_hash}_{cell_id}.webp")
                image.save(path)
            return True
        except Exception as e:
            print(f"Error saving compressed cropped image to disk cache: {e!s}")
            return False

    # Load compressed image from disk cache
    def load_compressed_images(self) -> list[ImageWithHash]:
        images = []
        for p in os.listdir(self.CACHE_DIR):
            path = join(self.CACHE_DIR, p)
            hash, ext = splitext(p)
            try:
                if isfile(path) and ext == ".webp":
                    image = Image.open(path)
                    images.append(ImageWithHash(image=image, hash=hash))
            except Exception as e:
                print(f"Error loading compressed image from disk cache: {e!s}; skipping")
        return images
