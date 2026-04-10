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
    // Pro Defaults
    useRsLoRA: false,
    useGradientCheckpointing: 'unsloth',
    targetModules: '',
    rewardType: 'none',
    hubUsername: '',
  });

  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [showExportModal, setShowExportModal] = useState<any>(null);

  const trainingOptions: { id: TrainingType; label: string }[] = [
    { id: 'LLMSFT', label: 'LLM Fine-tuning (SFT)' },
    { id: 'VisionSFT', label: 'Vision Fine-tuning' },
    { id: 'TTSSFT', label: 'TTS Fine-tuning' },
    { id: 'STTSFT', label: 'STT Fine-tuning' },
    { id: 'Embedding', label: 'Embedding Fine-tuning' },
    { id: 'OCR', label: 'OCR Fine-tuning' },
    { id: 'CPT', label: 'Continual Pretraining' },
    { id: 'GRPO', label: 'Reasoning (GRPO)' },
  ];

  const [mainTab, setMainTab] = useState<'training' | 'models' | 'settings' | 'editor'>('training');
  
  // Editor State
  const [fileTree, setFileTree] = useState<any[]>([]);
  const [openTabs, setOpenTabs] = useState<any[]>([]);
  const [activeTabPath, setActiveTabPath] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (mainTab === 'editor' && fileTree.length === 0) {
      fetchFileTree();
    }
  }, [mainTab]);

  const fetchFileTree = async () => {
    try {
      const res = await fetch('/api/files/tree');
      const data = await res.json();
      setFileTree(data);
    } catch (err) {
      console.error('Failed to fetch tree', err);
    }
  };

  const openFile = async (path: string, name: string) => {
    // Check if already open
    const existing = openTabs.find(t => t.path === path);
    if (existing) {
      setActiveTabPath(path);
      return;
    }

    try {
      const res = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      const { content } = await res.json();
      const newTab = { path, name, content, originalContent: content, isModified: false };
      setOpenTabs([...openTabs, newTab]);
      setActiveTabPath(path);
    } catch (err) {
      console.error('Failed to open file', err);
    }
  };

  const closeTab = (e: React.MouseEvent, path: string) => {
    e.stopPropagation();
    const newTabs = openTabs.filter(t => t.path !== path);
    setOpenTabs(newTabs);
    if (activeTabPath === path) {
      setActiveTabPath(newTabs.length > 0 ? newTabs[newTabs.length - 1].path : null);
    }
  };

  const handleEditorChange = (content: string) => {
    setOpenTabs(openTabs.map(t => 
      t.path === activeTabPath 
        ? { ...t, content, isModified: content !== t.originalContent } 
        : t
    ));
  };

  const saveActiveFile = async () => {
    const tab = openTabs.find(t => t.path === activeTabPath);
    if (!tab || isSaving) return;

    setIsSaving(true);
    try {
      const res = await fetch('/api/files/content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: tab.path, content: tab.content }),
      });
      if (res.ok) {
        setOpenTabs(openTabs.map(t => 
          t.path === activeTabPath 
            ? { ...t, originalContent: t.content, isModified: false } 
            : t
        ));
      }
    } catch (err) {
      console.error('Save failed', err);
    } finally {
      setIsSaving(false);
    }
  };

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
    ],
    'GRPO': [
      'mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit',
      'mlx-community/Qwen2.5-7B-Instruct-4bit',
      'mlx-community/Llama-3.1-8B-Instruct-4bit'
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
    } else if (type === 'GRPO') {
      setConfig(prev => ({ ...prev, modelName: defaultModel, datasetName: 'ServiceNow/RLHF-Reasoning-Prompt', rewardType: 'combined' }));
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

  const handleExport = async (action: 'hub' | 'gguf') => {
    if (isRunning) return;
    setIsRunning(true);
    setLogs('');
    setShowTerminal(true);
    setIsMinimized(false);
    setShowExportModal(null);

    try {
      const response = await fetch('/api/models/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action, 
          modelName: showExportModal?.path,
          hubUsername: config.hubUsername 
        }),
      });

      if (!response.body) throw new Error('No response body');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        setLogs(prev => prev + decoder.decode(value));
      }
    } catch (err: any) {
      setLogs(prev => prev + `\nExport failed: ${err.message}\n`);
    } finally {
      setIsRunning(false);
    }
  };

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
        <div className={`slim-icon ${mainTab === 'models' ? 'active' : ''}`} onClick={() => setMainTab('models')} title="Model Library">
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M22 19V5a2 2 0 00-2-2H4a2 2 0 00-2 2v14a2 2 0 002 2h16a2 2 0 002-2zM6 13h4M6 17h8M6 9h12"/></svg>
        </div>
        <div className={`slim-icon ${mainTab === 'editor' ? 'active' : ''}`} onClick={() => setMainTab('editor')} title="Code & Data Editor">
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M16 18l6-6-6-6M8 6l-6 6 6 6"/></svg>
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

                {activeTab === 'GRPO' && (
                  <div className="mac-form-group" style={{ marginTop: '16px', borderTop: '1px solid #333', paddingTop: '16px' }}>
                    <label className="mac-label">Reward Logic (Reasoning)</label>
                    <select name="rewardType" className="mac-input mac-select" value={config.rewardType} onChange={handleChange}>
                      <option value="accuracy">Accuracy Based</option>
                      <option value="format">Format Based (Reasoning Tags)</option>
                      <option value="combined">Combined (70% Acc / 30% Format)</option>
                      <option value="none">Custom Python Reward</option>
                    </select>
                    <p style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 8 }}>
                      GRPO requires a reward function to score model completions for reasoning traces.
                    </p>
                  </div>
                )}

                {/* Advanced Configuration Accordion */}
                <div style={{ marginTop: '24px', borderTop: '1px solid #333', paddingTop: '16px' }}>
                  <div 
                    onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                    style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center', 
                      cursor: 'pointer',
                      padding: '8px 4px',
                      color: isAdvancedOpen ? 'var(--accent)' : 'inherit'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 21a9 9 0 100-18 9 9 0 000 18z"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                      <h3 style={{ margin: 0, fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Performance & Experimental</h3>
                    </div>
                    <svg 
                      width="12" height="12" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24"
                      style={{ transform: isAdvancedOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}
                    >
                      <polyline points="6 9 12 15 18 9"/>
                    </svg>
                  </div>

                  {isAdvancedOpen && (
                    <div style={{ marginTop: 16, animation: 'fadeIn 0.2s ease-out' }}>
                      <div className="grid-2">
                        <div className="toggle-group" onClick={() => setConfig(prev => ({ ...prev, useRsLoRA: !prev.useRsLoRA }))}>
                          <input type="checkbox" className="mac-checkbox" checked={config.useRsLoRA} readOnly />
                          <span className="toggle-label">Use rsLoRA (Improved Stability)</span>
                        </div>
                        <div className="mac-form-group">
                          <label className="mac-label">Gradient Checkpointing</label>
                          <select 
                            name="useGradientCheckpointing" 
                            className="mac-input mac-select" 
                            value={config.useGradientCheckpointing === 'unsloth' ? 'unsloth' : config.useGradientCheckpointing ? 'true' : 'false'}
                            onChange={(e) => {
                              const val = e.target.value;
                              setConfig(prev => ({ 
                                ...prev, 
                                useGradientCheckpointing: val === 'unsloth' ? 'unsloth' : val === 'true' 
                              }));
                            }}
                          >
                            <option value="false">Off</option>
                            <option value="true">On (Standard)</option>
                            <option value="unsloth">On (Unsloth Optimized)</option>
                          </select>
                        </div>
                      </div>

                      <div className="mac-form-group" style={{ marginTop: 12 }}>
                        <label className="mac-label">Custom Target Modules (Optional)</label>
                        <input 
                          type="text" 
                          name="targetModules" 
                          className="mac-input" 
                          placeholder="e.g. q_proj, v_proj (Leave empty for defaults)"
                          value={config.targetModules} 
                          onChange={handleChange} 
                        />
                      </div>
                    </div>
                  )}
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
                  
                  <div style={{ display: 'flex', gap: 8, marginTop: 20 }}>
                    <button 
                      className="run-btn" 
                      style={{ flex: 1, fontSize: 13, padding: '10px' }}
                      onClick={() => useModel(model)}
                    >
                      Load
                    </button>
                    <button 
                      className="run-btn" 
                      style={{ 
                        flex: 1, 
                        fontSize: 13, 
                        padding: '10px', 
                        background: 'transparent', 
                        border: '1px solid #444',
                        color: '#999'
                      }}
                      onClick={() => setShowExportModal(model)}
                    >
                      Export
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Export Modal Overlays */}
            {showExportModal && (
              <div style={{ 
                position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, 
                background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
              }}>
                <div 
                  className="mac-card-container" 
                  style={{ width: 400, border: '1px solid var(--accent)', boxShadow: '0 20px 40px rgba(0,0,0,0.5)' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
                    <h3 style={{ margin: 0 }}>Model Export & Hub</h3>
                    <button onClick={() => setShowExportModal(null)} style={{ background: 'none', border: 'none', color: '#999', cursor: 'pointer' }}>×</button>
                  </div>
                  
                  <div className="mac-form-group">
                    <label className="mac-label">Hugging Face Username</label>
                    <input 
                      type="text" 
                      className="mac-input" 
                      placeholder="e.g. Viz38"
                      value={config.hubUsername}
                      onChange={(e) => setConfig(prev => ({ ...prev, hubUsername: e.target.value }))}
                    />
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 20 }}>
                    <button 
                      className="run-btn" 
                      style={{ fontSize: 12 }}
                      onClick={() => handleExport('hub')}
                    >
                      Push to Hub
                    </button>
                    <button 
                      className="run-btn" 
                      style={{ fontSize: 12, background: 'transparent', border: '1px solid #444' }}
                      onClick={() => handleExport('gguf')}
                    >
                      Export GGUF
                    </button>
                  </div>
                </div>
              </div>
            )}
            {scannedModels.length === 0 && (
              <div style={{ textAlign: 'center', padding: '100px 0', color: 'var(--text-dim)' }}>
                No local models detected. Make sure Ollama or LM Studio is configured correctly.
              </div>
            )}
          </div>
        </div>
      ) : mainTab === 'editor' ? (
        <div className="main-content">
          <div className="top-nav">
             <div className="top-nav-title">KNOWLEDGE & DATA EDITOR</div>
          </div>
          <div className="editor-layout">
            <div className="file-sidebar">
              <div style={{ padding: '0 16px 12px', fontSize: 11, color: 'var(--text-dim)', borderBottom: '1px solid var(--border)' }}>WORKSPACE</div>
              <div style={{ marginTop: 12 }}>
                {fileTree.map(item => (
                  <FileTreeItem key={item.id} item={item} onOpen={openFile} activePath={activeTabPath} />
                ))}
              </div>
            </div>
            <div className="editor-main">
              <div className="tab-bar">
                {openTabs.map(tab => (
                  <div 
                    key={tab.path} 
                    className={`tab ${activeTabPath === tab.path ? 'active' : ''}`}
                    onClick={() => setActiveTabPath(tab.path)}
                  >
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {tab.name}{tab.isModified ? '*' : ''}
                    </span>
                    <span className="tab-close" onClick={(e) => closeTab(e, tab.path)}>×</span>
                  </div>
                ))}
              </div>
              {activeTabPath ? (
                <>
                  <textarea 
                    className="code-area"
                    value={openTabs.find(t => t.path === activeTabPath)?.content || ''}
                    onChange={(e) => handleEditorChange(e.target.value)}
                    spellCheck={false}
                  />
                  <div className="editor-footer">
                    <button 
                      className="run-btn" 
                      style={{ padding: '8px 20px', fontSize: 12 }}
                      disabled={!openTabs.find(t => t.path === activeTabPath)?.isModified || isSaving}
                      onClick={saveActiveFile}
                    >
                      {isSaving ? 'Saving...' : 'Save Changes'}
                    </button>
                  </div>
                </>
              ) : (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
                  Select a file from the sidebar to start editing
                </div>
              )}
            </div>
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

const FileTreeItem = ({ item, level = 0, onOpen, activePath }: any) => {
  const [isOpen, setIsOpen] = useState(false);
  const isSelected = activePath === item.id;

  if (item.isDir) {
    return (
      <div>
        <div 
          className="file-item" 
          style={{ paddingLeft: 16 + level * 16 }}
          onClick={() => setIsOpen(!isOpen)}
        >
          <span className="folder-icon">{isOpen ? '▼' : '▶'}</span>
          <span>{item.name}</span>
        </div>
        {isOpen && item.children.map((child: any) => (
          <FileTreeItem key={child.id} item={child} level={level + 1} onOpen={onOpen} activePath={activePath} />
        ))}
      </div>
    );
  }

  const getIcon = (name: string) => {
    if (name.endsWith('.py')) return '🐍';
    if (name.endsWith('.json') || name.endsWith('.jsonl')) return '{}';
    if (name.endsWith('.md')) return 'M↓';
    return '📄';
  };

  return (
    <div 
      className={`file-item ${isSelected ? 'active' : ''}`} 
      style={{ paddingLeft: 16 + level * 16 }}
      onClick={() => onOpen(item.id, item.name)}
    >
      <span className="file-icon">{getIcon(item.name)}</span>
      <span>{item.name}</span>
    </div>
  );
};
