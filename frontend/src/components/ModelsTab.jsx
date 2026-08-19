import React, { useState, useEffect } from 'react';

export default function ModelsTab({ showError, setGlobalInfo }) {
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [loadingModel, setLoadingModel] = useState(null);
  const [fetching, setFetching] = useState(true);

  const fetchState = async (retries = 3) => {
    setFetching(true);
    try {
      const [modRes, statRes] = await Promise.all([
        fetch('http://localhost:8000/api/models'),
        fetch('http://localhost:8000/api/model/status')
      ]);
      if (modRes.ok) setModels(await modRes.json());
      if (statRes.ok) {
        const statData = await statRes.json();
        setActiveModel(statData.is_loaded ? statData.active_model_id : null);
      }
    } catch (err) {
      if (retries > 0) {
        console.log("Backend not ready, retrying in 1.5s...");
        setTimeout(() => fetchState(retries - 1), 1500);
        return;
      }
      showError("Failed to fetch models from backend. Ensure backend is running.");
    } finally {
      if (retries === 0 || !fetching) setFetching(false);
    }
  };

  useEffect(() => {
    fetchState();
  }, []);

  const loadModel = async (modelId) => {
    setLoadingModel(modelId);
    setGlobalInfo(`Loading ${modelId} into VRAM. This may take a minute...`);
    try {
      const res = await fetch('http://localhost:8000/api/model/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to load model.");
      }
      setActiveModel(modelId);
      setGlobalInfo(`Successfully loaded ${modelId}! Ready for inference.`);
    } catch (err) {
      showError(err.message);
      setGlobalInfo(null);
    } finally {
      setLoadingModel(null);
    }
  };

  return (
    <div>
      <div className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="font-headline text-4xl text-on-surface mb-2 tracking-tight">Model Library</h1>
          <p className="font-body text-on-surface-variant max-w-2xl text-sm">Select and load a Vision-Language Model into your GPU memory before running inference.</p>
        </div>
        <button onClick={fetchState} disabled={fetching} className="px-4 py-2 bg-surface-container border border-outline-variant/60 rounded-lg font-body text-sm font-medium hover:bg-surface-variant transition-colors app-shadow flex items-center gap-2">
          <span className={`material-symbols-outlined text-[18px] ${fetching ? 'animate-spin' : ''}`}>sync</span> Refresh
        </button>
      </div>

      {!activeModel ? (
        <div className="bg-surface-container-low border border-outline-variant/60 rounded-xl p-12 text-center app-shadow">
          <span className="material-symbols-outlined text-4xl text-on-surface-variant mb-4">memory</span>
          <h2 className="font-headline text-xl font-bold text-on-surface mb-2">No Model Loaded</h2>
          <p className="font-body text-sm text-on-surface-variant max-w-md mx-auto">
            Select a Vision-Language Model from your local library below to load it into GPU memory. 
            Only one model can be active at a time to prevent Out-Of-Memory errors.
          </p>
        </div>
      ) : (
        <div className="bg-primary/10 border border-primary/30 rounded-xl p-6 app-shadow flex items-center gap-6">
          <div className="bg-primary text-on-primary w-12 h-12 rounded-full flex items-center justify-center">
            <span className="material-symbols-outlined text-2xl">neurology</span>
          </div>
          <div>
            <h2 className="font-headline text-lg font-bold text-on-surface">GPU Active</h2>
            <p className="font-body text-sm text-on-surface-variant">
              Currently running: <span className="font-bold text-primary">{activeModel}</span>
            </p>
          </div>
        </div>
      )}

      <div className="mt-8">
        <h3 className="font-headline text-lg font-bold text-on-surface mb-4">Local Weights Library</h3>
        {models.length === 0 ? (
          <div className="text-center py-8 text-on-surface-variant font-body text-sm">
            No models found in the <code className="bg-surface-container px-1 py-0.5 rounded">weights/</code> directory.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map(m => {
              const isActive = activeModel === m.id;
              const isLoading = loadingModel === m.id;
              const isAnotherLoading = loadingModel !== null && !isLoading;
              
              return (
                <div key={m.id} className={`bg-surface-container-lowest rounded-xl p-6 app-shadow transition-all border ${isActive ? 'border-primary ring-1 ring-primary' : 'border-outline-variant/40 hover:border-outline/60'}`}>
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-2">
                      <span className={`material-symbols-outlined ${isActive ? 'text-primary' : 'text-on-surface-variant'}`}>
                        {isActive ? 'check_circle' : 'deployed_code'}
                      </span>
                      <h3 className="font-headline text-lg font-bold text-on-surface break-all">{m.id}</h3>
                    </div>
                    {isActive && <span className="bg-primary/10 text-primary font-body text-xs px-2 py-1 rounded-md font-bold uppercase tracking-wider">Active</span>}
                  </div>
                  <p className="font-mono text-xs text-on-surface-variant mb-6 break-all line-clamp-2" title={m.path}>
                    {m.path}
                  </p>
                  <button 
                    onClick={() => loadModel(m.id)}
                    disabled={isActive || isLoading || isAnotherLoading}
                    className={`w-full py-2.5 rounded-lg font-body text-sm font-bold flex items-center justify-center gap-2 transition-all ${
                      isActive ? 'bg-surface-variant text-on-surface-variant cursor-default' :
                      isLoading ? 'bg-primary/70 text-on-primary cursor-wait' :
                      isAnotherLoading ? 'bg-surface-container text-on-surface-variant opacity-50 cursor-not-allowed' :
                      'bg-primary text-on-primary hover:bg-primary/90'
                    }`}
                  >
                    {isLoading ? (
                      <><svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Loading...</>
                    ) : isActive ? (
                      'Currently Loaded'
                    ) : (
                      'Load to GPU'
                    )}
                  </button>
                </div>
              );
          })}
        </div>
      )}
      </div>
    </div>
  );
}
