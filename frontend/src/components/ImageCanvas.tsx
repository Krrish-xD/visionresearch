import { useState, useEffect, useRef } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Text, Group, Circle, Line } from 'react-konva';
import { AnalysisResult, DetectedObject, BoundingBox } from '../types/analysis';
import { OverlayControls } from './OverlayControls';
import styles from './ImageCanvas.module.css';

interface ImageCanvasProps {
  imageUrl: string;
  analysisResult: Partial<AnalysisResult>;
  hoveredObjectId?: string | null;
  activeOverlays: Record<string, boolean>;
  onOverlayChange: (key: string, value: boolean) => void;
}

export function ImageCanvas({ imageUrl, analysisResult, hoveredObjectId, activeOverlays, onOverlayChange }: ImageCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [depthImage, setDepthImage] = useState<HTMLImageElement | null>(null);
  const [segImage, setSegImage] = useState<HTMLImageElement | null>(null);
  
  // Viewport scaling and panning state
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  // Load image
  useEffect(() => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      setImage(img);
      fitImageToContainer(img.width, img.height);
    };
  }, [imageUrl]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && image) {
        fitImageToContainer(image.width, image.height);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [image]);

  // Load overlay images
  useEffect(() => {
    if (analysisResult.depth_map_path) {
      const img = new Image();
      // Use absolute path pointing to backend API
      img.src = `http://localhost:8000${analysisResult.depth_map_path}`;
      img.onload = () => setDepthImage(img);
    }
  }, [analysisResult.depth_map_path]);

  useEffect(() => {
    if (analysisResult.segmentation_map_path) {
      const img = new Image();
      img.src = `http://localhost:8000${analysisResult.segmentation_map_path}`;
      img.onload = () => setSegImage(img);
    }
  }, [analysisResult.segmentation_map_path]);

  const fitImageToContainer = (imgWidth: number, imgHeight: number) => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;
    const { clientWidth, clientHeight } = container;
    
    setDimensions({ width: clientWidth, height: clientHeight });
    
    const scaleX = clientWidth / imgWidth;
    const scaleY = clientHeight / imgHeight;
    // Fit within container with some padding
    const initialScale = Math.min(scaleX, scaleY) * 0.95;
    
    setScale(initialScale);
    
    // Center the image
    setPosition({
      x: (clientWidth - imgWidth * initialScale) / 2,
      y: (clientHeight - imgHeight * initialScale) / 2
    });
  };

  const handleWheel = (e: any) => {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    const oldScale = stage.scaleX();
    
    const mousePointTo = {
      x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
      y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
    };

    const newScale = e.evt.deltaY < 0 ? oldScale * 1.1 : oldScale / 1.1;
    // Clamp scale
    const clampedScale = Math.max(0.1, Math.min(newScale, 10));
    
    setScale(clampedScale);
    setPosition({
      x: -(mousePointTo.x - stage.getPointerPosition().x / clampedScale) * clampedScale,
      y: -(mousePointTo.y - stage.getPointerPosition().y / clampedScale) * clampedScale,
    });
  };

  // Render helpers
  const renderBoundingBoxes = () => {
    if (!activeOverlays.objects || !analysisResult.objects || !image) return null;

    return analysisResult.objects.map((obj, i) => {
      const { x_min, y_min, x_max, y_max } = obj.bbox;
      const x = x_min * image.width;
      const y = y_min * image.height;
      const w = (x_max - x_min) * image.width;
      const h = (y_max - y_min) * image.height;
      
      const isHovered = hoveredObjectId === `obj-${i}`;
      const color = '#3B82F6'; // Module color for YOLO

      return (
        <Group key={`obj-${i}`}>
          <Rect
            x={x}
            y={y}
            width={w}
            height={h}
            stroke={color}
            strokeWidth={isHovered ? 4 / scale : 2 / scale}
            fill={isHovered ? `${color}33` : 'transparent'} // 20% opacity fill on hover
          />
          {isHovered && (
            <Group x={x} y={y - (24 / scale)}>
              <Rect
                x={0}
                y={0}
                width={(obj.label.length * 8 + 40) / scale}
                height={20 / scale}
                fill={color}
                cornerRadius={4 / scale}
              />
              <Text
                x={4 / scale}
                y={4 / scale}
                text={`${obj.label} ${(obj.confidence * 100).toFixed(0)}%`}
                fontSize={12 / scale}
                fill="#ffffff"
                fontFamily="Inter"
              />
            </Group>
          )}
        </Group>
      );
    });
  };

  const renderFaces = () => {
    if (!activeOverlays.faces || !analysisResult.faces || !image) return null;

    return analysisResult.faces.map((face, i) => {
      const { x_min, y_min, x_max, y_max } = face.bbox;
      const x = x_min * image.width;
      const y = y_min * image.height;
      const w = (x_max - x_min) * image.width;
      const h = (y_max - y_min) * image.height;
      
      const color = '#EC4899'; // Module color for Faces

      return (
        <Rect
          key={`face-${i}`}
          x={x}
          y={y}
          width={w}
          height={h}
          stroke={color}
          strokeWidth={2 / scale}
          dash={[5 / scale, 5 / scale]}
        />
      );
    });
  };

  const renderPoses = () => {
    if (!activeOverlays.pose || !analysisResult.poses || !image) return null;

    const POSE_CONNECTIONS = [
      ['nose', 'left_eye'], ['left_eye', 'left_ear'],
      ['nose', 'right_eye'], ['right_eye', 'right_ear'],
      ['left_shoulder', 'right_shoulder'],
      ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
      ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
      ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
      ['left_hip', 'right_hip'],
      ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
      ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
    ];

    return analysisResult.poses.map((pose, idx) => {
      const kpDict: Record<string, any> = {};
      pose.keypoints.forEach(kp => {
        kpDict[kp.name] = { 
          x: kp.x * image.width, 
          y: kp.y * image.height,
          conf: kp.confidence 
        };
      });

      const color = '#10B981';

      return (
        <Group key={`pose-${idx}`}>
          {POSE_CONNECTIONS.map(([kp1, kp2], cIdx) => {
            const p1 = kpDict[kp1];
            const p2 = kpDict[kp2];
            if (p1 && p2 && p1.conf > 0.3 && p2.conf > 0.3) {
              return (
                <Line
                  key={`line-${idx}-${cIdx}`}
                  points={[p1.x, p1.y, p2.x, p2.y]}
                  stroke={color}
                  strokeWidth={3 / scale}
                  opacity={0.8}
                />
              );
            }
            return null;
          })}
          
          {pose.keypoints.filter(kp => kp.confidence > 0.3).map((kp, kpIdx) => (
            <Circle
              key={`kp-${idx}-${kpIdx}`}
              x={kp.x * image.width}
              y={kp.y * image.height}
              radius={4 / scale}
              fill="#ffffff"
              stroke={color}
              strokeWidth={2 / scale}
            />
          ))}
        </Group>
      );
    });
  };

  const renderDepthMap = () => {
    if (!activeOverlays.depth || !depthImage || !image) return null;
    return (
      <KonvaImage
        image={depthImage}
        width={image.width}
        height={image.height}
        opacity={0.8}
      />
    );
  };

  const renderSegmentation = () => {
    if (!activeOverlays.segmentation || !segImage || !image) return null;
    return (
      <KonvaImage
        image={segImage}
        width={image.width}
        height={image.height}
        opacity={0.6}
      />
    );
  };

  return (
    <div className={`glass-panel ${styles.canvasContainer}`} ref={containerRef}>
      {dimensions.width > 0 && image && (
        <Stage
          width={dimensions.width}
          height={dimensions.height}
          onWheel={handleWheel}
          scaleX={scale}
          scaleY={scale}
          x={position.x}
          y={position.y}
          draggable
          className={styles.stage}
        >
          {/* Base Image Layer */}
          <Layer listening={false}>
            <KonvaImage image={image} />
          </Layer>
          
          {/* Overlay Layer */}
          <Layer listening={false}>
            {renderDepthMap()}
            {renderSegmentation()}
            {renderBoundingBoxes()}
            {renderFaces()}
            {renderPoses()}
          </Layer>
        </Stage>
      )}
      
      {/* Overlay toggle controls wrapper */}
      <div className={styles.controlsWrapper}>
        <OverlayControls 
          activeOverlays={activeOverlays} 
          onChange={onOverlayChange} 
        />
      </div>
    </div>
  );
}
