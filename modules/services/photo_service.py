from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
from datetime import datetime
import hashlib
from typing import Dict, Optional
from pathlib import Path


class PhotoService:
    """Service for handling photo metadata extraction and basic operations."""

    def __init__(self, photos_directory: str = "photos"):
        self.photos_directory = Path(photos_directory)
        self.supported_formats = (
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".bmp",
            ".gif",
            ".webp",
        )

    def extract_metadata(self, image_path: Path) -> Optional[Dict]:
        """Extract comprehensive metadata from an image."""
        try:
            image_path = image_path.resolve()
            if not image_path.exists():
                print(f"File does not exist: {image_path}")
                return None

            file_size = image_path.stat().st_size

            with Image.open(image_path) as img:
                metadata = {
                    "filename": image_path.name,
                    "filepath": str(image_path),
                    "size_bytes": file_size,
                    "dimensions": f"{img.width}x{img.height}",
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "aspect_ratio": (
                        round(img.width / img.height, 2) if img.height > 0 else 0
                    ),
                    "file_hash": self._get_file_hash(image_path),
                }

                exifdata = img.getexif()
                if exifdata:
                    for tag_id, value in exifdata.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata[f"exif_{tag}"] = str(value)

                stat = image_path.stat()
                metadata["created_date"] = datetime.fromtimestamp(stat.st_ctime)
                metadata["modified_date"] = datetime.fromtimestamp(stat.st_mtime)

                return metadata

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def _get_file_hash(self, filepath: Path) -> str:
        """Generate hash for duplicate detection."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_all_metadata(self) -> pd.DataFrame:
        """Extract metadata from all photos in directory."""
        if not self.photos_directory.exists():
            print(f"Photos directory {self.photos_directory} does not exist!")
            return pd.DataFrame()

        all_metadata = []

        for file_path in self.photos_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_formats
            ):
                metadata = self.extract_metadata(file_path)
                if metadata:
                    all_metadata.append(metadata)

        return pd.DataFrame(all_metadata)

    def get_photo_count(self) -> int:
        """Get total count of photos in directory."""
        if not self.photos_directory.exists():
            return 0

        count = 0
        for file_path in self.photos_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_formats
            ):
                count += 1
        return count

    def validate_photos_directory(self) -> bool:
        """Check if photos directory exists and contains photos."""
        if not self.photos_directory.exists():
            return False

        for file_path in self.photos_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_formats
            ):
                return True

        return False
