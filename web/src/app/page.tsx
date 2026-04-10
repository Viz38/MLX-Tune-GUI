"use client";

import React, { useState, useRef, useEffect } from 'react';
import { TrainingType, MLXTuneConfig } from '@/lib/types';

interface NumberInputProps {
  label: string;
  name: string;
  value: number;
  onChange: (e: any) => void;
  min?: number;
  max?: number;
  step?: number;
}

const NumberInput: React.FC<NumberInputProps> = ({ label, name, value, onChange, min, max, step = 1 }) => {
  const adjust = (delta: number) => {
    const newValue = Number(value) + delta;
    if (min !== undefined && newValue < min) return;
    if (max !== undefined && newValue > max) return;
    onChange({ target: { name, value: newValue, type: 'number' } });
  };

  return (
    <div className="mac-form-group">
      <label className="mac-label">{label}</label>
      <div className="mac-number-input-wrapper">
        <input 
          type="number" 
          name={name} 
          className="mac-input" 
          value={value} 
          onChange={onChange} 
          min={min} 
          max={max} 
          step={step}
        />
        <div className="mac-spinner-container">
          <button className="mac-spinner-btn" onClick={() => adjust(step)} title="Increase">
            <svg width="10" height="10" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24"><polyline points="18 15 12 9 6 15"/></svg>
          </button>
          <button className="mac-spinner-btn" onClick={() => adjust(-step)} title="Decrease">
            <svg width="10" height="10" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default function Home() {
  const [activeTab, setActiveTab] = useState<TrainingType>('LLMSFT');
  const [logs, setLogs] = useState<string>('');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  const [scannedModels, setScannedModels] = useState<any[]>([]);
  const [modelSource, setModelSource] = useState<string>('hub');
  const [modelSearch, setModelSearch] = useState('');
  
  const [settings, setSettings] = useState({
    defaultOutputDir: 'outputs',
    pythonPath: 'python3',
    autoClearLogs: false,
    confirmBeforeRun: true,
  });

  const [config, setConfig] = useState<Partial<MLXTuneConfig>>({
    modelName: 'mlx-community/Llama-3.2-1B-Instruct-4bit',
    datasetName: 'yahma/alpaca-cleaned',
    datasetSplit: 'train[:100]',
    loadIn4Bit: true,
    loraRank: 16,
    loraAlpha: 16,
    batchSize: 2,
    learningRate: '2e-4',
    maxSteps: 50,
    mergeAndExport: false,
    outputDir: 'outputs',
    // Specialized Defaults
    finetuneVisionLayers: true,
    finetuneLanguageLayers: true,
    samplingRate: 24000,
    language: 'en',
    sttTask: 'transcribe',
    poolingStrategy: 'mean',
    lossType: 'infonce',
    temperature: 0.05,
    embeddingLearningRate: '5e-6',
    includeEmbeddings: true,
  });

  const trainingOptions: { id: TrainingType; label: string }[] = [
    { id: 'LLMSFT', label: 'LLM Fine-tuning (SFT)' },
    { id: 'VisionSFT', label: 'Vision Fine-tuning' },
    { id: 'TTSSFT', label: 'TTS Fine-tuning' },
    { id: 'STTSFT', label: 'STT Fine-tuning' },
    { id: 'Embedding', label: 'Embedding Fine-tuning' },
    { id: 'OCR', label: 'OCR Fine-tuning' },
    { id: 'CPT', label: 'Continual Pretraining' },
  ];

  // Load Settings
  useEffect(() => {
    const saved = localStorage.getItem('mlxtune_settings');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSettings(prev => ({ ...prev, ...parsed }));
        // Apply default output dir if config is still default
        setConfig(c => ({ ...c, outputDir: parsed.defaultOutputDir || c.outputDir }));
      } catch (e) {}
    }
  }, []);

  // Save Settings
  useEffect(() => {
    localStorage.setItem('mlxtune_settings', JSON.stringify(settings));
  }, [settings]);

  // Fetch local models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/models/scan');
        const data = await res.json();
        if (data.models) {
          setScannedModels(data.models);
        }
      } catch (e) {
        console.error("Failed to scan models", e);
      }
    };
    fetchModels();
  }, []);

  const POPULAR_HUB_MODELS: Record<string, string[]> = {
    'LLMSFT': [
      'mlx-community/Llama-3.2-1B-Instruct-4bit',
      'mlx-community/Llama-3.2-3B-Instruct-4bit',
      'mlx-community/Qwen2.5-1.5B-Instruct-4bit',
      'mlx-community/gemma-2-2b-it-4bit',
      'mlx-community/Phi-3.5-mini-instruct-4bit',
      'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
      'mlx-community/DeepSeek-V3-4bit'
    ],
    'VisionSFT': [
      'mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
      'mlx-community/Pixtral-12B-2409-4bit',
      'mlx-community/Llama-3.2-11B-Vision-Instruct-4bit',
      'mlx-community/PaliGemma-3B-pt-224-4bit'
    ],
    'TTSSFT': [
      'mlx-community/orpheus-3b-0.1-ft-bf16',
      'mlx-community/Llama-OuteTTS-1.0-1B-8bit',
      'mlx-community/Spark-TTS-0.5B-bf16'
    ],
    'STTSFT': [
      'mlx-community/whisper-tiny-asr-fp16',
      'mlx-community/whisper-large-v3-mlx',
      'mlx-community/distil-whisper-large-v3'
    ],
    'Embedding': [
      'mlx-community/all-MiniLM-L6-v2-bf16',
      'mlx-community/ModernBERT-base-mlx'
    ],
    'OCR': [
      'mlx-community/DeepSeek-OCR-8bit',
      'mlx-community/Qwen2.5-VL-3B-Instruct-4bit'
    ],
    'CPT': [
      'mlx-community/Llama-3.2-1B-4bit',
      'mlx-community/Qwen2.5-1.5B-4bit'
    ]
  };

  const filteredModels = modelSource === 'hub' 
    ? (POPULAR_HUB_MODELS[activeTab] || []).map(m => ({ name: m, path: m, source: 'hub' }))
    : scannedModels.filter(m => {
        if (modelSource === 'custom') return false;
        return m.source === modelSource;
      });

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : 
              type === 'number' ? Number(value) : value
    }));
  };

  const handleSourceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const source = e.target.value;
    setModelSource(source);
    if (source === 'hub') {
      const first = (POPULAR_HUB_MODELS[activeTab] || [])[0];
      setConfig(prev => ({ ...prev, modelName: first }));
    } else {
      // Find first model in that source as default
      const first = scannedModels.find(m => m.source === source);
      if (first) {
        setConfig(prev => ({ ...prev, modelName: first.path }));
      } else if (source !== 'custom') {
        setConfig(prev => ({ ...prev, modelName: '' }));
      }
    }
  };

  const setDefaultsForType = (type: TrainingType) => {
    setActiveTab(type);
    const models = POPULAR_HUB_MODELS[type] || [];
    const defaultModel = models[0] || '';
    
    if (type === 'VisionSFT') {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'HuggingFaceM4/VQAv2' }));
    } else if (type === 'TTSSFT') {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'MrDragonFox/Elise' }));
    } else if (type === 'STTSFT') {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'mozilla-foundation/common_voice_11_0' }));
    } else if (type === 'Embedding') {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'sentence-transformers/all-nli' }));
    } else if (type === 'OCR') {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'nielsr/cord-v2' }));
    } else {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'yahma/alpaca-cleaned' }));
    }
  };

  const [terminalHeight, setTerminalHeight] = useState(260);
  const [isMinimized, setIsMinimized] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);
  const [isResizing, setIsResizing] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newHeight = window.innerHeight - e.clientY;
      if (newHeight > 36 && newHeight < window.innerHeight * 0.8) {
        setTerminalHeight(newHeight);
        if (isMinimized && newHeight > 50) setIsMinimized(false);
      }
    };

    const handleMouseUp = () => setIsResizing(false);
    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, isMinimized]);

  const handleRun = async () => {
    if (isRunning) return;
    setIsRunning(true);
    setLogs('');
    setShowTerminal(true);
    setIsMinimized(false);

    try {
      const runConfig = { ...config, type: activeTab } as MLXTuneConfig;
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(runConfig),
      });

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        setLogs(prev => prev + text);
      }
    } catch (err: any) {
      setLogs(prev => prev + `\nFailed to start process: ${err.message}\n`);
    } finally {
      setIsRunning(false);
    }
  };

  const [mainTab, setMainTab] = useState<'training' | 'models' | 'settings'>('training');
  const revealModel = async (path: string) => {
    try {
      await fetch('/api/reveal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
    } catch (e) {
      console.error("Failed to reveal folder", e);
    }
  };

  const useModel = (model: any) => {
    setModelSource(model.source === 'hf' ? 'hub' : model.source);
    setConfig(prev => ({ ...prev, modelName: model.path }));
    setMainTab('training');
  };

  return (
    <div className="app-container">
      {/* Icon Sidebar - Slim */}
      <div className="slim-sidebar">
        <div 
          className={`slim-icon ${mainTab === 'training' ? 'active' : ''}`} 
          title="Training Orchestrator"
          onClick={() => setMainTab('training')}
        >
          <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5l10-5l-10-5zM2 17l10 5l10-5M2 12l10 5l10-5"/></svg>
        </div>
        <div 
          className={`slim-icon ${mainTab === 'models' ? 'active' : ''}`} 
          title="Model Library"
          onClick={() => setMainTab('models')}
        >
          <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
        </div>
        <div 
          className={`slim-icon ${mainTab === 'settings' ? 'active' : ''}`} 
          title="Global Settings"
          onClick={() => setMainTab('settings')}
        >
          <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
        </div>
      </div>

      {mainTab === 'training' ? (
        <>
          <div className="sidebar">
            <div className="sidebar-header">
              <h2>Training Types</h2>
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {trainingOptions.map(opt => (
                <div 
                  key={opt.id} 
                  className={`sidebar-item ${activeTab === opt.id ? 'active' : ''}`}
                  onClick={() => setDefaultsForType(opt.id)}
                >
                  {opt.label}
                </div>
              ))}
            </div>
          </div>

          <div className="main-content">
            <div className="top-nav">
              <div className="top-nav-title">MLX-Tune GUI</div>
            </div>
            <div className="section-header">
              <h1>{trainingOptions.find(o => o.id === activeTab)?.label}</h1>
            </div>
            <div className="scrollable-content">

              <div className="mac-card-container">
                <h3>Base Model Configuration</h3>
                <div className="grid-2">
                  <div className="mac-form-group">
                    <label className="mac-label">Model Source</label>
                    <select className="mac-input mac-select" value={modelSource} onChange={handleSourceChange}>
                        <option value="hub">Hugging Face Hub</option>
                        <option value="ollama">Ollama Local</option>
                        <option value="lmstudio">LM Studio Cache</option>
                        <option value="custom">Custom Local Path</option>
                    </select>
                  </div>
                  <div className="mac-form-group">
                    <label className="mac-label">Model Name / Path</label>
                    {modelSource === 'custom' ? (
                      <input 
                        type="text" 
                        name="modelName" 
                        className="mac-input" 
                        placeholder="/absolute/path/to/model"
                        value={config.modelName} 
                        onChange={handleChange} 
                      />
                    ) : (
                      <select name="modelName" className="mac-input mac-select" value={config.modelName} onChange={handleChange}>
                        {filteredModels.map(m => (
                          <option key={`${m.source}-${m.path}`} value={m.path}>{m.name}</option>
                        ))}
                      </select>
                    )}
                  </div>
                </div>

                <div className="mac-form-group" style={{ marginTop: '16px' }}>
                  <label className="mac-label">Dataset (Path or Hub ID)</label>
                  <input type="text" name="datasetName" className="mac-input" value={config.datasetName} onChange={handleChange} />
                </div>
              </div>

              <div className="mac-card-container">
                <h3>Fine-tuning Parameters</h3>
                <div className="grid-2">
                  <div className="toggle-group" onClick={() => setConfig(prev => ({ ...prev, loadIn4Bit: !prev.loadIn4Bit }))}>
                    <input type="checkbox" className="mac-checkbox" checked={config.loadIn4Bit} readOnly />
                    <span className="toggle-label">Quantize to 4-bit (Saves VRAM)</span>
                  </div>
                  <div className="toggle-group" onClick={() => setConfig(prev => ({ ...prev, mergeAndExport: !prev.mergeAndExport }))}>
                    <input type="checkbox" className="mac-checkbox" checked={config.mergeAndExport} readOnly />
                    <span className="toggle-label">Merge adapters on completion</span>
                  </div>
                </div>

                <div className="grid-2" style={{ marginTop: '16px', gridTemplateColumns: 'repeat(4, 1fr)' }}>
                  <NumberInput label="Batch Size" name="batchSize" value={config.batchSize || 1} onChange={handleChange} min={1} />
                  <NumberInput label="Max Steps" name="maxSteps" value={config.maxSteps || 50} onChange={handleChange} min={1} />
                  <NumberInput label="LoRA Rank" name="loraRank" value={config.loraRank || 16} onChange={handleChange} min={1} />
                  <NumberInput label="LoRA Alpha" name="loraAlpha" value={config.loraAlpha || 16} onChange={handleChange} min={1} />
                </div>
              </div>
            </div>

            <div className="action-bar">
              <button className="run-btn" onClick={handleRun} disabled={isRunning}>
                {isRunning ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div className="spinner" /> Training...
                  </div>
                ) : 'Begin Training'}
              </button>
            </div>

            {showTerminal && (
              <div 
                className={`terminal-container ${isMinimized ? 'minimized' : ''}`} 
                style={{ height: isMinimized ? 36 : terminalHeight }}
              >
                <div 
                  className="terminal-resizer" 
                  onMouseDown={() => setIsResizing(true)}
                />
                <div className="terminal-header">
                  <div className="terminal-tabs">
                    <div className="terminal-tab" style={{ color: 'var(--accent)', fontWeight: '600' }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: isRunning ? 'var(--terminal-green)' : '#444' }} />
                      CONSOLE
                    </div>
                    <div className="terminal-tab">
                      <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
                      generated_script.py
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <button 
                      onClick={() => setIsMinimized(!isMinimized)} 
                      style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', padding: 4, display: 'flex', alignItems: 'center' }}
                      title={isMinimized ? "Expand" : "Minimize"}
                    >
                      {isMinimized ? (
                        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polyline points="18 15 12 9 6 15"/></svg>
                      ) : (
                        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
                      )}
                    </button>
                    <div className="dot red" onClick={() => setShowTerminal(false)} style={{ cursor: 'pointer' }}></div>
                    <div className="dot yellow"></div>
                    <div className="dot green"></div>
                  </div>
                </div>
                <div className="mac-terminal-body">
                  {logs || 'Ready to fine-tune. Awaiting command...'}
                </div>
              </div>
            )}
          </div>
        </>
      ) : mainTab === 'models' ? (
        <div className="main-content">
          <div className="top-nav">
            <div className="top-nav-title">MLX-Tune GUI</div>
          </div>
          <div className="section-header">
            <h1>Local Weights</h1>
            <div style={{ position: 'relative', width: 300 }}>
              <input 
                type="text" 
                className="mac-input" 
                placeholder="Search models..." 
                style={{ paddingLeft: 40 }}
                value={modelSearch}
                onChange={(e) => setModelSearch(e.target.value)}
              />
              <svg style={{ position: 'absolute', left: 14, top: 14, color: 'var(--text-dim)' }} width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            </div>
          </div>
          <div className="scrollable-content">

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 20 }}>
              {scannedModels
                .filter(m => m.name.toLowerCase().includes(modelSearch.toLowerCase()))
                .map(model => (
                <div key={`${model.source}-${model.path}`} className="mac-card-container" style={{ margin: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                      <div style={{ 
                        fontSize: 10, 
                        fontWeight: 700, 
                        textTransform: 'uppercase', 
                        padding: '2px 6px', 
                        background: model.source === 'hf' ? '#FF9D00' : model.source === 'ollama' ? '#fff' : '#0a84ff',
                        color: model.source === 'ollama' ? '#000' : '#fff',
                        borderRadius: 4
                      }}>
                        {model.source}
                      </div>
                      <button 
                        onClick={() => revealModel(model.path)}
                        style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer' }}
                        title="Reveal in Finder"
                      >
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
                      </button>
                    </div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: '#fff', marginBottom: 4, wordBreak: 'break-all' }}>{model.name}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-dim)', wordBreak: 'break-all', opacity: 0.7 }}>{model.path}</div>
                  </div>
                  
                  <button 
                    className="run-btn" 
                    style={{ marginTop: 20, width: '100%', fontSize: 13, padding: '10px' }}
                    onClick={() => useModel(model)}
                  >
                    Load in Orchestrator
                  </button>
                </div>
              ))}
            </div>
            {scannedModels.length === 0 && (
              <div style={{ textAlign: 'center', padding: '100px 0', color: 'var(--text-dim)' }}>
                No local models detected. Make sure Ollama or LM Studio is configured correctly.
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="main-content">
          <div className="top-nav">
            <div className="top-nav-title">MLX-Tune GUI</div>
          </div>
          <div className="section-header">
            <h1>Global Preferences</h1>
          </div>
          <div className="scrollable-content">
            <div className="grid-2">
              <div className="mac-card-container">
                <h3>Training Environment</h3>
                <div className="mac-form-group">
                  <label className="mac-label">Default Output Directory</label>
                  <input 
                    type="text" 
                    className="mac-input" 
                    value={settings.defaultOutputDir}
                    onChange={(e) => setSettings(prev => ({ ...prev, defaultOutputDir: e.target.value }))}
                  />
                </div>
                <div className="mac-form-group">
                  <label className="mac-label">Python Interpreter Path</label>
                  <input 
                    type="text" 
                    className="mac-input" 
                    value={settings.pythonPath}
                    onChange={(e) => setSettings(prev => ({ ...prev, pythonPath: e.target.value }))}
                  />
                  <p style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 8 }}>Recommended: /usr/bin/python3 or path to your venv</p>
                </div>
              </div>

              <div className="mac-card-container">
                <h3>Interface & Safety</h3>
                <div className="mac-form-group">
                  <div className="toggle-group" onClick={() => setSettings(prev => ({ ...prev, autoClearLogs: !prev.autoClearLogs }))}>
                    <input type="checkbox" className="mac-checkbox" checked={settings.autoClearLogs} readOnly />
                    <span className="toggle-label">Auto-clear console on new run</span>
                  </div>
                  <div className="toggle-group" style={{ marginTop: 12 }} onClick={() => setSettings(prev => ({ ...prev, confirmBeforeRun: !prev.confirmBeforeRun }))}>
                    <input type="checkbox" className="mac-checkbox" checked={settings.confirmBeforeRun} readOnly />
                    <span className="toggle-label">Show confirmation before training</span>
                  </div>
                </div>
              </div>
            </div>

            <div style={{ opacity: 0.5, fontSize: 12, color: 'var(--text-dim)', marginTop: 40 }}>
              MLX-Tune GUI v0.4.20<br/>
              Running on Apple Silicon (Unified Memory)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
