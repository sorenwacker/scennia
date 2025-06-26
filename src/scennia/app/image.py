import base64
import hashlib
import io

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile

from scennia.app.data import (
    EncodedImage,
)


# Decode image from base64
def decode_image(contents: str) -> ImageFile:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))


# Encode image to base64
def encode_image(image: Image.Image) -> EncodedImage:
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="WebP")
    buffer.seek(0)
    # Convert to base64
    contents = f"data:image/webp;base64,{base64.b64encode(buffer.read()).decode()}"
    return EncodedImage(contents=contents, width=image.width, height=image.height)


# Calculate image hash for caching
def calculate_image_hash(image: ImageFile) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


# Create a cell crop, returning an uncompressed cropped image of the cell.
def crop_cell(image: ImageFile, bbox: list[int], padding=10) -> Image.Image:
    # Get bounding box with padding
    y0, x0, y1, x1 = bbox
    padding = 10
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(image.height, y1 + padding)
    x1 = min(image.width, x1 + padding)

    # Create a cropped version
    image_array = np.asarray(image)
    img_cropped = image_array[y0:y1, x0:x1]
    return Image.fromarray(img_cropped)
