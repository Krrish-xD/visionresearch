import React, { useState, useRef, useEffect } from 'react';

export default function VisualizerTab({ showError, setGlobalInfo }) {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState("https://lh3.googleusercontent.com/aida-public/AB6AXuA_KaNU2r7obrvrgHL2bP08fcJzbRbfMtyDGFDBvdrULoAz8jfWim81zMv8jb4F5-0b-gz7Wy8QbKAGnBwIrv5EEQ8t0d4U1Os8Rw3usdJ66wTnHt-oivMADsIn0gcyPGUsA4va-JVsUxsCNrGDiFMq3J30JJr95WRr7-wwBmaR5DxE9OHLT6FSTNWEwM3Inj7woKfGPe4t6ADvFtVeGIgd_A4qv2M36Bb4mOKMavCXVFHJT5ALkDuC");
  const [prompt, setPrompt] = useState("Analyze the architectural composition, focusing on the interplay of natural light and geometric shadows in this modern space.");
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    temperature: 1.0,
    top_p: 1.0,
    top_k: 50,
    max_tokens: 100
  });
  
  const [tokens, setTokens] = useState([]);
  const [stats, setStats] = useState({ conf: null, count: null, latency: null });
  const [fullResponse, setFullResponse] = useState(null);
  
  const [tooltip, setTooltip] = useState({ visible: false, token: '', logprob: '', conf: '', top: 0, left: 0, color: '' });
  
  const fileInputRef = useRef(null);
  const wrapperRef = useRef(null);

  const fetchModels = async (retries = 3) => {
    setIsRefreshing(true);
    try {
      const [modRes, statRes] = await Promise.all([
        fetch('http://localhost:8000/api/models'),
        fetch('http://localhost:8000/api/model/status')
      ]);
      
      let defaultModel = '';
      if (statRes.ok) {
        const statData = await statRes.json();
        if (statData.is_loaded) defaultModel = statData.active_model_id;
      }

      if (modRes.ok) {
        const data = await modRes.json();
        setModels(data);
        if (data.length > 0) {
          if (defaultModel && data.find(m => m.id === defaultModel)) {
            setSelectedModel(defaultModel);
          } else {
            setSelectedModel(data[0].id);
          }
        } else {
          setSelectedModel('');
        }
      }
    } catch (err) {
      if (retries > 0) {
        console.log("Backend not ready, retrying in 1.5s...");
        setTimeout(() => fetchModels(retries - 1), 1500);
        return;
      }
      console.error("Failed to fetch models:", err);
    } finally {
      if (retries === 0 || !isRefreshing) {
        setTimeout(() => setIsRefreshing(false), 500);
      }
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleImageUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (event) => setImagePreview(event.target.result);
      reader.readAsDataURL(file);
    }
  };

  const clearWorkspace = () => {
    setTokens([]);
    setStats({ conf: null, count: null, latency: null });
    setFullResponse(null);
  };

  const runInference = async () => {
    if (!selectedModel) {
      showError("Please select a model from the dropdown first.");
      return;
    }
    
    if (!imageFile && !imagePreview.startsWith('data:')) {
      try {
        const res = await fetch(imagePreview);
        const blob = await res.blob();
        const file = new File([blob], "default.jpg", { type: "image/jpeg" });
        setImageFile(file);
      } catch (e) {
        showError("Please upload a local image first.");
        return;
      }
    }

    setLoading(true);
    setTokens([]);
    
    const formData = new FormData();
    formData.append('image', imageFile || await (await fetch(imagePreview)).blob());
    formData.append('prompt', prompt);
    formData.append('model_id', selectedModel);
    formData.append('temperature', settings.temperature);
    formData.append('top_p', settings.top_p);
    formData.append('top_k', settings.top_k);
    formData.append('max_tokens', settings.max_tokens);

    const startTime = performance.now();

    try {
      const response = await fetch('http://localhost:8000/api/generate', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Inference failed.');
      }

      const data = await response.json();
      setFullResponse(data);
      setTokens(data.tokens);
      
      const latency = ((performance.now() - startTime) / 1000).toFixed(1);
      const avgProb = data.tokens.reduce((acc, curr) => acc + curr.prob_percent, 0) / data.tokens.length;
      
      setStats({
        conf: avgProb,
        count: data.tokens.length,
        latency: latency
      });

    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleMouseEnterToken = (e, token) => {
    const rect = e.target.getBoundingClientRect();
    const wrapperRect = wrapperRef.current.getBoundingClientRect();
    
    let color = '#c0392b';
    if (token.prob_percent > 80) color = '#27ae60';
    else if (token.prob_percent > 30) color = '#f1c40f';

    setTooltip({
      visible: true,
      token: JSON.stringify(token.text),
      logprob: token.logprob.toFixed(4),
      conf: token.prob_percent.toFixed(2) + '%',
      top: rect.top - wrapperRect.top + wrapperRef.current.scrollTop - 80,
      left: rect.left - wrapperRect.left + wrapperRef.current.scrollLeft + (rect.width / 2),
      color: color
    });
  };

  const handleMouseLeaveToken = () => {
    setTooltip(prev => ({ ...prev, visible: false }));
  };

  const exportJSON = () => {
    if (!fullResponse) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(fullResponse, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = "logprobs_export.json";
    a.click();
  };

  return (
    <div>
      <div className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="font-headline text-4xl text-on-surface mb-2 tracking-tight">Image Visualizer</h1>
          <p className="font-body text-on-surface-variant max-w-2xl text-sm">Upload an image and provide a descriptive prompt to generate a token heatmap analysis of the model's confidence.</p>
        </div>
        <button onClick={clearWorkspace} className="px-4 py-2 bg-surface-container border border-outline-variant/60 rounded-lg font-body text-sm font-medium hover:bg-surface-variant transition-colors sahara-shadow">
          Clear Workspace
        </button>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 h-[calc(100vh-280px)] min-h-[600px]">
        
        {/* Left Column */}
        <div className="xl:col-span-7 flex flex-col gap-6">
          <div className="bg-surface-container-lowest rounded-xl sahara-shadow sahara-border p-6 flex flex-col h-[400px]">
            <h3 className="font-headline text-xl mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">image</span> Source Image
            </h3>
            <input type="file" ref={fileInputRef} onChange={handleImageUpload} className="hidden" accept="image/*" />
            <div onClick={() => fileInputRef.current.click()} className="flex-1 relative rounded-lg overflow-hidden border border-outline-variant/40 group bg-surface-container cursor-pointer">
              <img alt="Preview" className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" src={imagePreview} />
              <div className="absolute inset-0 bg-inverse-surface/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                <button className="bg-surface-bright/90 backdrop-blur text-on-surface px-4 py-2 rounded-lg font-body text-sm font-medium flex items-center gap-2 hover:bg-surface-bright">
                  <span className="material-symbols-outlined text-[18px]">upload</span> Replace Image
                </button>
              </div>
            </div>
          </div>

          <div className="bg-surface-container-lowest rounded-xl sahara-shadow sahara-border p-6 flex flex-col gap-5">
            <div>
              <label className="block font-headline text-lg mb-2 text-on-surface">Instruction Prompt</label>
              <textarea 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full bg-surface-container-low border border-outline-variant/60 rounded-lg p-4 font-body text-sm text-on-surface focus:ring-1 focus:ring-primary focus:border-primary transition-all resize-none h-28" 
              />
            </div>
            <div className="flex items-end gap-4">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <label className="block font-body text-xs font-medium text-on-surface-variant uppercase tracking-wider">Vision Model</label>
                  <div className="flex items-center gap-2">
                    <button onClick={() => setShowSettings(true)} className="text-xs text-on-surface-variant hover:text-primary transition-colors flex items-center gap-1" title="Generation Settings">
                      <span className="material-symbols-outlined text-[16px]">settings</span>
                    </button>
                    <button onClick={fetchModels} className="text-xs text-primary hover:underline flex items-center gap-1">
                      <span className={`material-symbols-outlined text-[14px] ${isRefreshing ? 'animate-spin' : ''}`}>refresh</span> Refresh
                    </button>
                  </div>
                </div>
                <div className="relative">
                  <select 
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-surface-container-low border border-outline-variant/60 rounded-lg py-2.5 pl-4 pr-10 font-body text-sm text-on-surface appearance-none focus:ring-1 focus:ring-primary focus:border-primary transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                    disabled={models.length === 0}
                  >
                    {models.length > 0 ? models.map(m => (
                      <option key={m.id} value={m.id}>{m.id}</option>
                    )) : (
                      <option value="" disabled>Awaiting Download...</option>
                    )}
                  </select>
                  <span className="material-symbols-outlined absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none">expand_more</span>
                </div>
              </div>
              <button 
                onClick={runInference} 
                disabled={loading || !selectedModel}
                className={`font-body text-sm font-bold py-2.5 px-6 rounded-lg transition-all flex items-center justify-center gap-2 h-[42px] min-w-[160px] ${(!selectedModel || loading) ? 'bg-surface-container-highest text-on-surface-variant cursor-not-allowed' : 'bg-primary text-on-primary hover:bg-primary/90'}`}
              >
                {loading ? (
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <><span className="material-symbols-outlined text-[18px]">{selectedModel ? 'play_arrow' : 'lock'}</span> {selectedModel ? 'Run Inference' : 'No Model'}</>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-5">
          <div className="bg-surface-container-lowest rounded-xl app-shadow app-border p-6 flex flex-col h-[400px]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-headline text-lg font-bold text-on-surface">Vision Input</h2>
              <button className="p-2 rounded-full hover:bg-surface-container transition-colors text-on-surface-variant tooltip-trigger">
                <span className="material-symbols-outlined text-[20px]">help</span>
              </button>
            </div>
              <div className="flex gap-2 text-xs font-body items-center bg-surface-container p-1.5 rounded-md border border-outline-variant/30">
                <span className="w-3 h-3 rounded-full bg-error/20 border border-error/40"></span><span className="text-on-surface-variant mr-2">Low</span>
                <span className="w-3 h-3 rounded-full bg-[#f1c40f]/20 border border-[#f1c40f]/40"></span><span className="text-on-surface-variant mr-2">Med</span>
                <span className="w-3 h-3 rounded-full bg-success/20 border border-success/40" style={{backgroundColor: 'rgba(39,174,96,0.2)', borderColor: 'rgba(39,174,96,0.4)'}}></span><span className="text-on-surface-variant">High</span>
              </div>

            <div ref={wrapperRef} className="p-6 flex-1 overflow-y-auto bg-surface-bright/50 relative min-h-[300px]">
              
              {tokens.length > 0 ? (
                <div className="font-body text-lg leading-loose flex flex-wrap gap-1.5 items-center relative z-10">
                  {tokens.map((tok, i) => {
                    let tClass = 'token-low';
                    if (tok.prob_percent > 80) tClass = 'token-high';
                    else if (tok.prob_percent > 30) tClass = 'token-med';
                    
                    return (
                      <span 
                        key={i} 
                        onMouseEnter={(e) => handleMouseEnterToken(e, tok)}
                        onMouseLeave={handleMouseLeaveToken}
                        className={`token-item px-1.5 py-0.5 rounded font-mono ${tClass}`}
                      >
                        {tok.text.replace(' ', ' ')}
                      </span>
                    )
                  })}
                </div>
              ) : (
                <div className="absolute inset-0 pointer-events-none opacity-5 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[150px]">data_object</span>
                </div>
              )}

              {/* Tooltip */}
              <div 
                className={`absolute z-50 bg-inverse-surface text-inverse-on-surface p-3 rounded-lg shadow-lg font-body text-xs w-48 pointer-events-none transition-opacity duration-150 ${tooltip.visible ? 'opacity-100' : 'opacity-0 hidden'}`}
                style={{ top: tooltip.top, left: tooltip.left, transform: 'translateX(-50%)' }}
              >
                <div className="flex justify-between border-b border-outline/30 pb-1 mb-1">
                  <span className="text-secondary-fixed-dim">Token:</span>
                  <span className="font-mono text-primary-fixed-dim">{tooltip.token}</span>
                </div>
                <div className="flex justify-between mb-0.5">
                  <span className="text-secondary-fixed-dim">Logprob:</span>
                  <span className="font-mono">{tooltip.logprob}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-fixed-dim">Confidence:</span>
                  <span className="font-mono" style={{color: tooltip.color}}>{tooltip.conf}</span>
                </div>
                <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-inverse-surface rotate-45 border-r border-b border-outline/20"></div>
              </div>

            </div>

            {/* Footer Stats */}
            <div className="bg-surface-container-low border-t border-outline-variant/40 p-4 flex justify-between items-center">
              <div className="flex gap-6">
                <div>
                  <span className="block font-body text-[10px] uppercase tracking-wider text-on-surface-variant">Avg Confidence</span>
                  <span className="font-mono text-sm" style={{color: stats.conf > 80 ? '#27ae60' : stats.conf > 30 ? '#f1c40f' : '#c0392b'}}>
                    {stats.conf ? `${stats.conf.toFixed(1)}%` : '-'}
                  </span>
                </div>
                <div>
                  <span className="block font-body text-[10px] uppercase tracking-wider text-on-surface-variant">Tokens Gen</span>
                  <span className="font-mono text-sm">{stats.count || '-'}</span>
                </div>
                <div>
                  <span className="block font-body text-[10px] uppercase tracking-wider text-on-surface-variant">Latency</span>
                  <span className="font-mono text-sm">{stats.latency ? `${stats.latency}s` : '-'}</span>
                </div>
              </div>
              <button onClick={exportJSON} disabled={!fullResponse} className="text-primary font-body text-xs font-medium flex items-center gap-1 hover:underline disabled:opacity-50">
                Export JSON <span className="material-symbols-outlined text-[14px]">download</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {showSettings && (
        <div className="fixed inset-0 bg-surface/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-surface-container-lowest border border-outline-variant/60 rounded-2xl p-6 w-full max-w-md app-shadow">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-headline text-xl font-bold text-on-surface flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">tune</span> 
                Generation Settings
              </h3>
              <button onClick={() => setShowSettings(false)} className="text-on-surface-variant hover:text-on-surface transition-colors">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            
            <div className="space-y-6">
              {/* Temperature */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="font-body text-sm font-bold text-on-surface">Temperature</label>
                  <input 
                    type="number" min="0" max="2" step="0.01" 
                    value={settings.temperature}
                    onChange={e => setSettings({...settings, temperature: parseFloat(e.target.value)})}
                    className="bg-surface-container border border-outline-variant/50 rounded-md px-2 py-1 text-sm w-20 text-right font-mono text-on-surface"
                  />
                </div>
                <input 
                  type="range" min="0" max="2" step="0.01" 
                  value={settings.temperature}
                  onChange={e => setSettings({...settings, temperature: parseFloat(e.target.value)})}
                  className="w-full accent-primary"
                />
                <p className="text-xs text-on-surface-variant mt-1">Controls randomness. 0 for greedy decoding.</p>
              </div>
              
              {/* Top P */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="font-body text-sm font-bold text-on-surface">Top P (Nucleus Sampling)</label>
                  <input 
                    type="number" min="0.01" max="1" step="0.01" 
                    value={settings.top_p}
                    onChange={e => setSettings({...settings, top_p: parseFloat(e.target.value)})}
                    className="bg-surface-container border border-outline-variant/50 rounded-md px-2 py-1 text-sm w-20 text-right font-mono text-on-surface"
                  />
                </div>
                <input 
                  type="range" min="0.01" max="1" step="0.01" 
                  value={settings.top_p}
                  onChange={e => setSettings({...settings, top_p: parseFloat(e.target.value)})}
                  className="w-full accent-primary"
                />
              </div>

              {/* Top K */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="font-body text-sm font-bold text-on-surface">Top K</label>
                  <input 
                    type="number" min="1" max="200" step="1" 
                    value={settings.top_k}
                    onChange={e => setSettings({...settings, top_k: parseInt(e.target.value)})}
                    className="bg-surface-container border border-outline-variant/50 rounded-md px-2 py-1 text-sm w-20 text-right font-mono text-on-surface"
                  />
                </div>
                <input 
                  type="range" min="1" max="200" step="1" 
                  value={settings.top_k}
                  onChange={e => setSettings({...settings, top_k: parseInt(e.target.value)})}
                  className="w-full accent-primary"
                />
              </div>

              {/* Max Tokens */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="font-body text-sm font-bold text-on-surface">Max Tokens</label>
                  <input 
                    type="number" min="1" max="2048" step="1" 
                    value={settings.max_tokens}
                    onChange={e => setSettings({...settings, max_tokens: parseInt(e.target.value)})}
                    className="bg-surface-container border border-outline-variant/50 rounded-md px-2 py-1 text-sm w-20 text-right font-mono text-on-surface"
                  />
                </div>
                <input 
                  type="range" min="1" max="2048" step="1" 
                  value={settings.max_tokens}
                  onChange={e => setSettings({...settings, max_tokens: parseInt(e.target.value)})}
                  className="w-full accent-primary"
                />
              </div>
            </div>
            
            <div className="mt-8 pt-4 border-t border-outline-variant/40 flex justify-end">
              <button 
                onClick={() => setShowSettings(false)}
                className="bg-primary text-on-primary px-6 py-2 rounded-lg font-body text-sm font-bold hover:bg-primary/90 transition-colors"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
