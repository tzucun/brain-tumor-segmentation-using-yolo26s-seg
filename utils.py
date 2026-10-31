from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image


def safe_filename(name: str, default: str = "file") -> str:
    base = Path(name).name.strip() or default
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._") or default
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(base).suffix.lower())
    return f"{stem}{suffix}"


def load_image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
