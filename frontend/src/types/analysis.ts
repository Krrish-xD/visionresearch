export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface DetectedObject {
  label: string;
  confidence: number;
  bbox: BoundingBox;
  area_fraction: number;
}

export interface TextRegion {
  content: string;
  confidence: number;
  bbox: BoundingBox;
  language?: string;
}

export interface FaceAnalysis {
  bbox: BoundingBox;
  age?: number;
  gender?: string;
  emotion?: string;
  emotion_confidence?: number;
}

export interface Keypoint {
  name: string;
  x: number;
  y: number;
  confidence: number;
}

export interface PoseEstimation {
  person_bbox: BoundingBox;
  keypoints: Keypoint[];
  confidence: number;
}

export interface ColorInfo {
  hex: string;
  rgb: [number, number, number];
  percentage: number;
  name: string;
}

export interface NSFWResult {
  is_nsfw: boolean;
  category: string;
  confidence: number;
}

export interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  file_size_bytes: number;
  mode: string;
  exif?: Record<string, any>;
  camera?: string;
  date_taken?: string;
  gps?: Record<string, any>;
}

export interface AnalysisResult {
  image_id: string;
  filename: string;
  timestamp: string;
  total_processing_time_ms: number;
  modules_executed: string[];
  schema_version: string;
  metadata?: ImageMetadata;
  caption?: string;
  detailed_description?: string;
  tags: string[];
  objects: DetectedObject[];
  text_regions: TextRegion[];
  faces: FaceAnalysis[];
  poses: PoseEstimation[];
  colors: ColorInfo[];
  nsfw?: NSFWResult;
  depth_map_path?: string;
  segmentation_map_path?: string;
  embedding?: number[];
  module_timings: Record<string, number>;
}

export type ModuleStatus = 'pending' | 'running' | 'complete' | 'error';

export interface ModuleEvent {
  module: string;
  status: ModuleStatus;
  results?: any;
  progress?: number;
  error?: string;
  display_name?: string;
  timing_ms?: number;
}

export interface ModuleState {
  name: string;
  display_name: string;
  status: ModuleStatus;
  results?: any;
  timing_ms?: number;
  error?: string;
  stage?: number;
}
