import React from 'react';
import { Target, Layers, Zap } from 'lucide-react';
import './RecommendationPanel.css';

const RecommendationPanel = () => {
    return (
        <div className="recommendation-container">
            <div className="recommendation-header">
                <h3>Choose Based on Deployment Priority</h3>
            </div>
            
            <div className="recommendation-cards">
                <div className="rec-card">
                    <div className="rec-icon-wrapper primary">
                        <Target size={24} />
                    </div>
                    <h4>HIGHEST ACCURACY</h4>
                    <span className="rec-model">Baseline ResNet-18</span>
                    <p>The uncompressed reference model remains the highest-accuracy benchmark result.</p>
                </div>
                <div className="rec-card">
                    <div className="rec-icon-wrapper primary">
                        <Target size={24} />
                    </div>
                    <h4>BEST OVERALL TRADE-OFF</h4>
                    <span className="rec-model">INT8 Quantization</span>
                    <p>Minimal accuracy loss (-0.36 pp) with significant memory footprint reduction.</p>
                </div>

                <div className="rec-card">
                    <div className="rec-icon-wrapper warning">
                        <Layers size={24} />
                    </div>
                    <h4>BALANCED COMPRESSION</h4>
                    <span className="rec-model">Pruning + Fine-tuning</span>
                    <p>Good trade-off between parameter reduction and latency without extreme quantization.</p>
                </div>

                <div className="rec-card">
                    <div className="rec-icon-wrapper success">
                        <Zap size={24} />
                    </div>
                    <h4>MAXIMUM EFFICIENCY</h4>
                    <span className="rec-model">Knowledge Distillation V2</span>
                    <p>Smallest footprint and fastest CPU inference for heavily constrained edge devices.</p>
                </div>
            </div>
        </div>
    );
};

export default RecommendationPanel;
