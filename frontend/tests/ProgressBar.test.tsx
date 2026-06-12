/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressBar } from '../src/components/ProgressBar';

// Minimal mock to test rendering logic
describe('ProgressBar Component', () => {
  
  it('renders nothing when not analyzing and progress is 0', () => {
    const { container } = render(
      <ProgressBar 
        progress={0} 
        isAnalyzing={false} 
        moduleStates={{}} 
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders progress text and bar when analyzing', () => {
    render(
      <ProgressBar 
        progress={45} 
        isAnalyzing={true} 
        moduleStates={{}} 
      />
    );
    
    // Check if the progress text is present
    expect(screen.getByText('Processing... 45%')).toBeDefined();
  });

  it('displays an error message if pipeline fails', () => {
    render(
      <ProgressBar 
        progress={10} 
        isAnalyzing={false} 
        moduleStates={{}} 
        error="GPU Out of Memory"
      />
    );
    
    expect(screen.getByText('Pipeline Error')).toBeDefined();
    expect(screen.getByText('GPU Out of Memory')).toBeDefined();
  });

  it('renders module tracker emojis correctly', () => {
    const mockStates = {
      'metadata': { name: 'metadata', display_name: 'Metadata', status: 'complete' as const },
      'object_detection': { name: 'object_detection', display_name: 'Objects', status: 'running' as const }
    };

    render(
      <ProgressBar 
        progress={50} 
        isAnalyzing={true} 
        moduleStates={mockStates} 
      />
    );
    
    // Check for the completed emoji
    expect(screen.getByText('✅')).toBeDefined();
    // Check for the running emoji
    expect(screen.getByText('🔄')).toBeDefined();
  });
});
