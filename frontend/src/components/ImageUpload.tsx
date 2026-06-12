import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon } from 'lucide-react';
import styles from './ImageUpload.module.css';

interface ImageUploadProps {
  onUpload: (file: File) => void;
  disabled?: boolean;
}

export function ImageUpload({ onUpload, disabled = false }: ImageUploadProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0 && !disabled) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload, disabled]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.gif']
    },
    maxFiles: 1,
    disabled
  });

  // Global paste handler
  React.useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      if (disabled) return;
      
      const items = e.clipboardData?.items;
      if (!items) return;

      for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
          const file = item.getAsFile();
          if (file) {
            onUpload(file);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [onUpload, disabled]);

  return (
    <div 
      {...getRootProps()} 
      className={`glass-panel ${styles.dropzone} ${isDragActive ? styles.active : ''} ${disabled ? styles.disabled : ''}`}
    >
      <input {...getInputProps()} />
      <div className={styles.content}>
        <div className={styles.iconWrapper}>
          {isDragActive ? (
            <Upload size={48} className={styles.icon} />
          ) : (
            <ImageIcon size={48} className={styles.icon} />
          )}
        </div>
        
        <h3 className={styles.title}>
          {isDragActive ? 'Drop image here...' : 'Upload an image'}
        </h3>
        
        <p className={styles.subtitle}>
          Drag and drop, click to select, or <kbd className={styles.kbd}>Ctrl+V</kbd> to paste
        </p>
        
        <div className={styles.formats}>
          Supports JPG, PNG, WebP, GIF
        </div>
      </div>
    </div>
  );
}
