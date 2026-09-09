import React from 'react';
import './ModelCard.css';

const getBestForLabel = (name) => {
    if (name.includes('INT8')) return 'Memory-efficient deployment';
    if (name.includes('Pruning')) return 'Balanced compression';
    if (name.includes('Knowledge')) return 'Lightweight inference';
    return 'Accuracy reference';
};

const getCategoryBadge = (name) => {
    if (name.includes('INT8')) return { text: 'QUANTIZED', class: 'badge-quantized' };
    if (name.includes('Pruning')) return { text: 'PRUNED', class: 'badge-pruned' };
    if (name.includes('Knowledge')) return { text: 'DISTILLED', class: 'badge-distilled' };
    return { text: 'BASELINE', class: 'badge-baseline' };
};

const ModelCard = ({ model, onClick }) => {
    const category = getCategoryBadge(model.name);

    return (
        <div className="model-card" onClick={() => onClick(model)}>
            <div className="model-card-header">
                <div className="model-header-top">
                    <h3>{model.name}</h3>
                    <span className={`category-badge ${category.class}`}>{category.text}</span>
                </div>
                <span className="best-for-label">{getBestForLabel(model.name)}</span>
            </div>
            
            <div className="model-metrics-grid">
                <div className="model-metric">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">{model.accuracy}</span>
                </div>
                <div className="model-metric">
                    <span className="metric-label">Size</span>
                    <span className="metric-value">{model.size}</span>
                </div>
                <div className="model-metric">
                    <span className="metric-label">Latency</span>
                    <span className="metric-value">{model.cpu_inference}</span>
                </div>
                <div className="model-metric">
                    <span className="metric-label">Reduction</span>
                    <span className={model.size_reduction !== "0.00%" ? "metric-value text-success" : "metric-value text-muted"}>
                        {model.size_reduction}
                    </span>
                </div>
            </div>
            
            <div className="model-card-footer">
                <span className="action-text">View Details →</span>
            </div>
        </div>
    );
};

export default ModelCard;
