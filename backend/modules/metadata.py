"""Metadata extraction module."""

import asyncio
from typing import Any

from PIL import Image, ExifTags

from core.base import BaseAnalyzer
from core.schemas import ImageMetadata


class MetadataAnalyzer(BaseAnalyzer):
    """Extracts EXIF data, image dimensions, and basic properties.
    
    This is a CPU-only module that runs instantly in Stage 0.
    """

    name = "metadata"
    display_name = "Image Metadata"
    estimated_vram_mb = 0
    requires_gpu = False
    stage = 0

    async def load_model(self, device: str = "cpu") -> None:
        # No model to load
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs: Any) -> dict:
        # Run CPU-bound PIL operations in threadpool to not block asyncio loop
        return await asyncio.to_thread(self._extract_metadata, image)

    def _extract_metadata(self, image: Image.Image) -> dict:
        width, height = image.size
        
        # Parse EXIF
        exif_data = {}
        camera_model = None
        date_taken = None
        gps_info = {}
        
        try:
            exif = image.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        # Skip huge binary blobs
                        continue
                        
                    exif_data[str(tag)] = str(value)
                    
                    if tag == "Model":
                        camera_model = str(value)
                    elif tag == "DateTimeOriginal" or tag == "DateTime":
                        date_taken = str(value)
                        
                # Extract GPS if present
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    for key, val in gps_ifd.items():
                        tag_name = ExifTags.GPSTAGS.get(key, key)
                        gps_info[str(tag_name)] = str(val)
        except Exception:
            pass  # Corrupted EXIF is common, just ignore

        metadata = ImageMetadata(
            width=width,
            height=height,
            format=image.format or "UNKNOWN",
            file_size_bytes=0,  # We don't have the raw bytes here easily, could pass from router
            mode=image.mode,
            exif=exif_data if exif_data else None,
            camera=camera_model,
            date_taken=date_taken,
            gps=gps_info if gps_info else None,
        )
        
        return {"metadata": metadata}

    async def unload_model(self) -> None:
        self._is_loaded = False
