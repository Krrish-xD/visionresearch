"""Pydantic schemas for all analysis output types.

These models define the structured output format for every analysis module.
Bounding boxes use normalized coordinates (0.0 to 1.0) relative to image dimensions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# =============================================================================
# Primitive Types
# =============================================================================


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates (0.0 to 1.0)."""

    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)


# =============================================================================
# Module-Specific Result Types
# =============================================================================


class DetectedObject(BaseModel):
    """A single detected object with bounding box."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    area_fraction: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of image area occupied"
    )


class TextRegion(BaseModel):
    """A region of text detected by OCR."""

    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    language: str | None = None


class FaceAnalysis(BaseModel):
    """Analysis results for a detected face."""

    bbox: BoundingBox
    age: int | None = None
    gender: str | None = None
    emotion: str | None = None
    emotion_confidence: float | None = Field(None, ge=0.0, le=1.0)


class Keypoint(BaseModel):
    """A single body keypoint (normalized coordinates)."""

    name: str
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class PoseEstimation(BaseModel):
    """Pose estimation result for a single person."""

    person_bbox: BoundingBox
    keypoints: list[Keypoint]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ColorInfo(BaseModel):
    """Information about a dominant color in the image."""

    hex: str
    rgb: tuple[int, int, int]
    percentage: float = Field(..., ge=0.0, le=1.0, description="Fraction of image")
    name: str = Field(..., description="Nearest named color")


class NSFWResult(BaseModel):
    """Content safety classification result."""

    is_nsfw: bool
    category: str = Field(..., description="safe, suggestive, or explicit")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ImageMetadata(BaseModel):
    """Technical metadata extracted from the image file."""

    width: int
    height: int
    format: str
    file_size_bytes: int
    mode: str = ""  # e.g., "RGB", "RGBA"
    exif: dict | None = None
    camera: str | None = None
    date_taken: str | None = None
    gps: dict | None = None


# =============================================================================
# Module Result Wrapper
# =============================================================================


class ModuleResult(BaseModel):
    """Result from a single analysis module."""

    module_name: str
    display_name: str
    success: bool = True
    error: str | None = None
    timing_ms: float = 0.0
    data: dict = Field(default_factory=dict)


# =============================================================================
# Complete Analysis Result
# =============================================================================


class AnalysisResult(BaseModel):
    """Complete analysis output for a single image.

    This is the top-level response model containing results from all modules.
    """

    # Meta
    image_id: str
    filename: str
    timestamp: str
    total_processing_time_ms: float = 0.0
    modules_executed: list[str] = Field(default_factory=list)
    schema_version: str = "1.0.0"

    # Results
    metadata: ImageMetadata | None = None
    caption: str | None = None
    detailed_description: str | None = None
    tags: list[str] = Field(default_factory=list)
    objects: list[DetectedObject] = Field(default_factory=list)
    text_regions: list[TextRegion] = Field(default_factory=list)
    faces: list[FaceAnalysis] = Field(default_factory=list)
    poses: list[PoseEstimation] = Field(default_factory=list)
    colors: list[ColorInfo] = Field(default_factory=list)
    nsfw: NSFWResult | None = None
    depth_map_path: str | None = None
    segmentation_map_path: str | None = None
    embedding: list[float] | None = None

    # Per-module timing
    module_timings: dict[str, float] = Field(default_factory=dict)

    # Raw module results (for extensibility)
    module_results: list[ModuleResult] = Field(default_factory=list)
