import { useState, useEffect, useRef } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Text, Group } from 'react-konva';
import { AnalysisResult, DetectedObject, BoundingBox } from '../types/analysis';
import styles from './ImageCanvas.module.css';

interface ImageCanvasProps {
  imageUrl: string;
  analysisResult: Partial<AnalysisResult>;
  hoveredObjectId?: string | null;
  activeOverlays: Record<string, boolean>;
}

export function ImageCanvas({ imageUrl, analysisResult, hoveredObjectId, activeOverlays }: ImageCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  
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
            {renderBoundingBoxes()}
            {renderFaces()}
            {/* Future: Depth Map, Segmentation Masks, Pose Skeletons */}
          </Layer>
        </Stage>
      )}
      
      {/* Overlay toggle controls wrapper */}
      <div className={styles.controlsWrapper}>
        {/* Render OverlayControls here via children/props in the future */}
      </div>
    </div>
  );
}
