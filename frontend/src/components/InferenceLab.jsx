import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle, AlertCircle, X, Loader2 } from 'lucide-react';
import { runInference } from '../services/inference';
import './InferenceLab.css';

const MODELS = [
    { id: 'baseline', name: 'Baseline ResNet-18', group: 'Reference' },
    { id: 'int8', name: 'INT8 Quantization', group: 'Compressed' },
    { id: 'pruned', name: 'Pruning + Fine-tuning', group: 'Compressed' },
    { id: 'kd', name: 'Knowledge Distillation V2', group: 'Compressed' }
];

const InferenceLab = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [selectedModel, setSelectedModel] = useState('baseline');
    
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    
    // For "Compare All"
    const [isComparing, setIsComparing] = useState(false);
    const [compareResults, setCompareResults] = useState([]);
    const [compareProgress, setCompareProgress] = useState(0);

    const fileInputRef = useRef(null);

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        if (!file.type.startsWith('image/')) {
            setError('Please upload a valid image file (PNG/JPG).');
            return;
        }

        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setError(null);
        setResult(null);
        setCompareResults([]);
    };

    const clearImage = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResult(null);
        setCompareResults([]);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleRunInference = async () => {
        if (!selectedFile) return;
        
        setIsLoading(true);
        setError(null);
        setResult(null);
        setCompareResults([]);

        try {
            const data = await runInference(selectedFile, selectedModel);
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleCompareAll = async () => {
        if (!selectedFile) return;
        
        setIsComparing(true);
        setError(null);
        setResult(null);
        setCompareResults([]);
        setCompareProgress(0);

        const results = [];
        for (let i = 0; i < MODELS.length; i++) {
            setCompareProgress(i + 1);
            try {
                const data = await runInference(selectedFile, MODELS[i].id);
                results.push({ ...data, success: true, modelName: MODELS[i].name });
            } catch (err) {
                results.push({ 
                    success: false, 
                    modelName: MODELS[i].name, 
                    error: err.message,
                    model_id: MODELS[i].id 
                });
            }
        }

        setCompareResults(results);
        setIsComparing(false);
    };

    return (
        <section id="inference" className="dashboard-section inference-lab">
            <div className="section-header">
                <div>
                    <h2 className="section-title">Live Model Inference</h2>
                    <p className="section-subtitle">Upload a CIFAR-100 image and evaluate the trained compression models in real-time.</p>
                </div>
            </div>

            <div className="inference-grid">
                {/* LEFT SIDE: Image Upload */}
                <div className="upload-panel panel-card">
                    <h3>Input Image</h3>
                    
                    {!previewUrl ? (
                        <div 
                            className="dropzone"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <UploadCloud className="drop-icon" size={48} />
                            <h4>Drag & Drop Image</h4>
                            <p>or click to browse</p>
                            <span className="file-hint">Supports PNG, JPG, JPEG</span>
                            <input 
                                type="file" 
                                ref={fileInputRef}
                                onChange={handleFileSelect}
                                accept="image/png, image/jpeg, image/jpg" 
                                style={{ display: 'none' }} 
                            />
                        </div>
                    ) : (
                        <div className="image-preview-container">
                            <div className="image-preview-wrapper">
                                <img src={previewUrl} alt="Upload preview" className="image-preview" />
                            </div>
                            <div className="image-meta">
                                <div className="meta-info">
                                    <ImageIcon size={14} />
                                    <span>{selectedFile?.name}</span>
                                </div>
                                <button className="btn-secondary btn-sm" onClick={clearImage}>
                                    <X size={14} /> Replace
                                </button>
                            </div>
                        </div>
                    )}
                    
                    <div className="disclaimer-note">
                        <AlertCircle size={14} className="text-warning" />
                        <p>Models are trained on CIFAR-100 (32×32 RGB). For meaningful predictions, upload CIFAR-100-style images.</p>
                    </div>
                </div>

                {/* RIGHT SIDE: Controls & Results */}
                <div className="controls-panel">
                    <div className="panel-card model-selector-card">
                        <h3>Model Selection</h3>
                        
                        <div className="model-options">
                            {MODELS.map(model => (
                                <label 
                                    key={model.id} 
                                    className={`model-radio-card ${selectedModel === model.id ? 'selected' : ''}`}
                                >
                                    <input 
                                        type="radio" 
                                        name="model_selection" 
                                        value={model.id}
                                        checked={selectedModel === model.id}
                                        onChange={() => setSelectedModel(model.id)}
                                    />
                                    <div className="radio-content">
                                        <span className="model-group">{model.group}</span>
                                        <span className="model-name">{model.name}</span>
                                    </div>
                                    <div className="radio-indicator"></div>
                                </label>
                            ))}
                        </div>

                        <div className="action-buttons">
                            <button 
                                className="btn-primary btn-lg" 
                                onClick={handleRunInference}
                                disabled={!selectedFile || isLoading || isComparing}
                            >
                                {isLoading ? (
                                    <><Loader2 className="spin" size={18} /> Running Inference...</>
                                ) : (
                                    <>Run Inference</>
                                )}
                            </button>
                            
                            <button 
                                className="btn-secondary btn-lg" 
                                onClick={handleCompareAll}
                                disabled={!selectedFile || isLoading || isComparing}
                            >
                                {isComparing ? (
                                    <><Loader2 className="spin" size={18} /> Testing {compareProgress} / 4</>
                                ) : (
                                    <>Compare All Models</>
                                )}
                            </button>
                        </div>
                        
                        {error && (
                            <div className="error-alert">
                                <AlertCircle size={16} />
                                <span>{error}</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* SINGLE RESULT VIEW */}
            {result && !isComparing && compareResults.length === 0 && (
                <div className="result-panel panel-card fade-in">
                    <div className="result-header">
                        <h3>Inference Result</h3>
                        <span className="badge badge-success"><CheckCircle size={12} /> Success</span>
                    </div>
                    
                    <div className="result-metrics-grid">
                        <div className="r-metric highlight-metric">
                            <span className="r-label">Prediction</span>
                            <span className="r-value text-accent capitalize">{result.prediction}</span>
                        </div>
                        <div className="r-metric">
                            <span className="r-label">Prediction Confidence</span>
                            <span className="r-value">{result.confidence.toFixed(1)}%</span>
                        </div>
                        <div className="r-metric">
                            <span className="r-label">Live Inference Time</span>
                            <span className="r-value text-warning">{result.inference_time_ms.toFixed(2)} ms</span>
                        </div>
                        <div className="r-metric">
                            <span className="r-label">Model Size</span>
                            <span className="r-value text-success">{result.model_size_mb}</span>
                        </div>
                    </div>
                    
                    <div className="result-benchmark-meta">
                        <span className="meta-label">Benchmark Context:</span>
                        <div className="meta-tags">
                            <span className="m-tag">Accuracy: {result.accuracy}</span>
                            <span className="m-tag">Size Reduction: {result.size_reduction}</span>
                            <span className="m-tag">Accuracy Drop: {result.accuracy_drop}</span>
                        </div>
                    </div>
                </div>
            )}

            {/* COMPARE ALL VIEW */}
            {compareResults.length > 0 && (
                <div className="compare-panel panel-card fade-in">
                    <div className="result-header">
                        <h3>Model Comparison on Same Input</h3>
                    </div>
                    
                    <div className="compare-table-wrapper">
                        <table className="compare-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Prediction</th>
                                    <th>Confidence</th>
                                    <th>Live Inference Time</th>
                                    <th>Model Size</th>
                                    <th>Size Reduction</th>
                                </tr>
                            </thead>
                            <tbody>
                                {compareResults.map((res, idx) => (
                                    <tr key={idx}>
                                        <td className="fw-600">{res.modelName}</td>
                                        {res.success ? (
                                            <>
                                                <td className="text-accent capitalize">{res.prediction}</td>
                                                <td>{res.confidence.toFixed(1)}%</td>
                                                <td className="text-warning">{res.inference_time_ms.toFixed(2)} ms</td>
                                                <td className="text-success">{res.model_size_mb}</td>
                                                <td>{res.size_reduction}</td>
                                            </>
                                        ) : (
                                            <td colSpan="5" className="error-cell">
                                                <AlertCircle size={14} /> Failed: {res.error}
                                            </td>
                                        )}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

        </section>
    );
};

export default InferenceLab;
