import React, { useState } from 'react';
import VisualizerTab from './components/VisualizerTab';
import ModelsTab from './components/ModelsTab';

export default function App() {
  const [currentTab, setCurrentTab] = useState('models');
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);

  const showError = (msg) => {
    setError(msg);
    setTimeout(() => setError(null), 5000);
  };
  
  const setGlobalInfo = (msg) => {
    setInfo(msg);
    if (msg) setTimeout(() => setInfo(null), 8000);
  };

  const clearInfo = () => setInfo(null);

  return (
    <div className="flex flex-col min-h-screen text-on-surface">
      {/* TopAppBar */}
      <header className="bg-surface-bright border-b border-outline-variant/60 shadow-sm w-full top-0 sticky z-50">
        <div className="flex items-center justify-between px-8 h-16 w-full max-w-full">
          <div className="flex items-center gap-8">
            <div className="font-headline text-2xl font-bold text-primary tracking-tight">Vision Research</div>
            <nav className="hidden md:flex items-center gap-6">
              <a onClick={e => e.preventDefault()} className="text-primary border-b-2 border-primary pb-1 font-body text-sm font-medium" href="#">Workbench</a>
              <a onClick={e => e.preventDefault()} className="text-on-surface-variant font-body text-sm font-medium hover:text-primary transition-colors" href="#">Datasets (WIP)</a>
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={e => e.preventDefault()} className="text-on-surface-variant hover:text-primary transition-colors"><span className="material-symbols-outlined">settings</span></button>
            <button onClick={e => e.preventDefault()} className="text-on-surface-variant hover:text-primary transition-colors"><span className="material-symbols-outlined">help</span></button>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* SideNavBar */}
        <aside className="hidden lg:flex flex-col bg-surface-container-low border-r border-outline-variant/60 h-[calc(100vh-64px)] sticky top-16 w-64 p-4 shrink-0">
          <div className="mb-6 px-2">
            <h2 className="font-headline text-xl text-on-surface">Project Workspace</h2>
            <p className="font-body text-xs text-on-surface-variant mt-1">V-Token v2.4</p>
          </div>
          <nav className="flex-1 space-y-2">
            <button 
              onClick={() => setCurrentTab('models')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg font-bold font-body text-sm transition-colors ${currentTab === 'models' ? 'bg-primary-container text-on-primary-container' : 'text-on-surface-variant hover:bg-surface-variant'}`}
            >
              <span className="material-symbols-outlined text-[20px]">folder_managed</span> Models Library
            </button>
            <button 
              onClick={() => setCurrentTab('visualizer')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg font-bold font-body text-sm transition-colors ${currentTab === 'visualizer' ? 'bg-primary-container text-on-primary-container' : 'text-on-surface-variant hover:bg-surface-variant'}`}
            >
              <span className="material-symbols-outlined text-[20px]">image_search</span> Visualizer
            </button>
          </nav>
        </aside>

        {/* Main Area */}
        <main className="flex-1 overflow-y-auto p-8 relative">
          
          {/* Error Toast */}
          <div className={`absolute top-4 left-1/2 -translate-x-1/2 bg-error-container text-on-error-container border border-error/20 px-4 py-3 rounded-lg shadow-md flex items-center gap-3 z-50 transition-opacity duration-300 ${error ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
            <span className="material-symbols-outlined text-error">error</span>
            <span className="font-body text-sm font-medium">{error}</span>
          </div>
          
          {/* Info Toast */}
          <div className={`absolute top-16 left-1/2 -translate-x-1/2 bg-surface-container-highest text-on-surface border border-outline-variant/40 px-4 py-2 rounded-lg shadow-sm flex items-center gap-3 z-40 transition-opacity duration-500 ${info ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
            <span className="material-symbols-outlined text-primary text-[18px]">info</span>
            <span className="font-body text-sm font-medium">{info}</span>
            <button onClick={clearInfo} className="ml-2 text-on-surface-variant hover:text-on-surface"><span className="material-symbols-outlined text-[16px]">close</span></button>
          </div>

          {currentTab === 'models' ? (
            <ModelsTab showError={showError} setGlobalInfo={setGlobalInfo} />
          ) : (
            <VisualizerTab showError={showError} setGlobalInfo={setGlobalInfo} />
          )}

        </main>
      </div>
    </div>
  );
}
