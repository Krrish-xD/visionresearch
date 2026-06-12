"""Color palette extraction module."""

import asyncio
from typing import Any

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import webcolors

from core.base import BaseAnalyzer
from core.schemas import ColorInfo


class ColorPaletteAnalyzer(BaseAnalyzer):
    """Extracts dominant colors from an image using k-means clustering.
    
    Runs in Stage 0 (CPU-only).
    """

    name = "colors"
    display_name = "Color Palette"
    estimated_vram_mb = 0
    requires_gpu = False
    stage = 0

    async def load_model(self, device: str = "cpu") -> None:
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self._extract_colors, image)

    def _closest_color_name(self, rgb: tuple[int, int, int]) -> str:
        """Find the nearest CSS color name."""
        try:
            return webcolors.rgb_to_name(rgb)
        except ValueError:
            # If no exact match, find nearest
            min_colors = {}
            try:
                hex_names = webcolors.CSS3_HEX_TO_NAMES
            except AttributeError:
                hex_names = webcolors._definitions._CSS3_HEX_TO_NAMES
            
            for hex_val, name in hex_names.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
                rd = (r_c - rgb[0]) ** 2
                gd = (g_c - rgb[1]) ** 2
                bd = (b_c - rgb[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]

    def _extract_colors(self, image: Image.Image, n_colors: int = 5) -> dict:
        # Resize to speed up k-means
        img_small = image.copy()
        img_small.thumbnail((150, 150))
        
        # Convert to numpy array of RGB pixels
        pixels = np.array(img_small).reshape(-1, 3)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(pixels)
        
        # Count frequencies
        counts = np.bincount(labels)
        total = len(pixels)
        
        # Sort by frequency
        order = np.argsort(counts)[::-1]
        
        colors = []
        for idx in order:
            rgb_arr = kmeans.cluster_centers_[idx].astype(int)
            rgb = (int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2]))
            hex_str = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            pct = float(counts[idx]) / total
            name = self._closest_color_name(rgb)
            
            colors.append(
                ColorInfo(
                    hex=hex_str,
                    rgb=rgb,
                    percentage=pct,
                    name=name
                )
            )
            
        return {"colors": colors}

    async def unload_model(self) -> None:
        self._is_loaded = False
