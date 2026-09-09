import React from 'react';
import { X, Activity, HardDrive, Cpu, Percent, BarChart } from 'lucide-react';
import './ModelDetail.css';

const ModelDetail = ({ model, baseline, onClose }) => {
    if (!model) return null;

    // Helper to calculate relative difference visually
    const getDiffElement = (currentStr, baselineStr, isHigherBetter = true) => {
        if (!baselineStr || !currentStr || model.name.includes('Baseline')) return null;
        
        const current = parseFloat(currentStr.replace(/[^0-9.-]+/g, ""));
        const base = parseFloat(baselineStr.replace(/[^0-9.-]+/g, ""));
        
        if (isNaN(current) || isNaN(base)) return null;

        const diff = current - base;
        const diffPercent = ((current - base) / base) * 100;
        
        let isPositiveEffect = isHigherBetter ? diff > 0 : diff < 0;
        let colorClass = isPositiveEffect ? 'text-success' : 'text-danger';
        if (Math.abs(diff) < 0.01) colorClass = 'text-secondary'; // Neutral

        let sign = diff > 0 ? '+' : '';
        return (
            <div className={`diff-badge ${colorClass}`}>
                {sign}{isHigherBetter ? diff.toFixed(2) + ' pp' : diffPercent.toFixed(2) + '%'}
            </div>
        );
    };

    return (
        <div className="model-detail-overlay" onClick={onClose}>
            <div className="model-detail-drawer" onClick={(e) => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>
                    <X size={24} />
                </button>
                
                <div className="drawer-header">
                    <h2>{model.name}</h2>
                    <span className="drawer-subtitle">Detailed Metrics</span>
                </div>

                <div className="drawer-content">
                    <div className="detail-metric-card">
                        <div className="detail-metric-header">
                            <Activity size={18} className="text-accent" />
                            <span>Accuracy</span>
                        </div>
                        <div className="detail-metric-value">
                            <h3>{model.accuracy}</h3>
                            {getDiffElement(model.accuracy, baseline?.accuracy, true)}
                        </div>
                    </div>

                    <div className="detail-metric-card">
                        <div className="detail-metric-header">
                            <HardDrive size={18} className="text-accent" />
                            <span>Model Size</span>
                        </div>
                        <div className="detail-metric-value">
                            <h3>{model.size}</h3>
                            {getDiffElement(model.size, baseline?.size, false)}
                        </div>
                    </div>

                    <div className="detail-metric-card">
                        <div className="detail-metric-header">
                            <Cpu size={18} className="text-accent" />
                            <span>CPU Inference</span>
                        </div>
                        <div className="detail-metric-value">
                            <h3>{model.cpu_inference}</h3>
                            {getDiffElement(model.cpu_inference, baseline?.cpu_inference, false)}
                        </div>
                    </div>

                    <div className="detail-metric-card">
                        <div className="detail-metric-header">
                            <BarChart size={18} className="text-accent" />
                            <span>Parameters</span>
                        </div>
                        <div className="detail-metric-value">
                            <h3>{model.parameters}</h3>
                        </div>
                    </div>

                    <div className="detail-metric-card">
                        <div className="detail-metric-header">
                            <Percent size={18} className="text-success" />
                            <span>Size Reduction</span>
                        </div>
                        <div className="detail-metric-value">
                            <h3 className="text-success">{model.size_reduction}</h3>
                        </div>
                    </div>

                    <div className="detail-metric-card">
                        <div className="detail-metric-header">
                            <Activity size={18} className="text-warning" />
                            <span>Accuracy Drop</span>
                        </div>
                        <div className="detail-metric-value">
                            <h3 className="text-warning">{model.accuracy_drop}</h3>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelDetail;
