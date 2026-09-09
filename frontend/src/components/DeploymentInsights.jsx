import React from 'react';
import { Cpu, HardDrive, Target, Zap } from 'lucide-react';
import './DeploymentInsights.css';

const DeploymentInsights = () => {
    return (
        <div className="insights-container">
            <div className="insights-header">
                <h3>Edge Deployment Readiness</h3>
                <p>Engineering analysis of benchmark results for edge device constraints.</p>
            </div>
            
            <div className="insights-grid">
                <div className="insight-card">
                    <div className="insight-icon-wrapper blue">
                        <HardDrive size={20} />
                    </div>
                    <div className="insight-content">
                        <h4>Memory Constraints</h4>
                        <p><strong>INT8</strong> and <strong>KD V2</strong> reduce model size by approximately 87%, making them ideal for microcontrollers or edge devices with tight VRAM limits.</p>
                    </div>
                </div>

                <div className="insight-card">
                    <div className="insight-icon-wrapper green">
                        <Zap size={20} />
                    </div>
                    <div className="insight-content">
                        <h4>Latency Optimizations</h4>
                        <p><strong>KD V2</strong> provides the lowest measured CPU inference time, offering the best real-time performance on lower-tier hardware.</p>
                    </div>
                </div>

                <div className="insight-card">
                    <div className="insight-icon-wrapper purple">
                        <Target size={20} />
                    </div>
                    <div className="insight-content">
                        <h4>Accuracy Retention</h4>
                        <p><strong>INT8</strong> retains accuracy closest to the baseline (-0.36 pp), making it the safest choice when precision cannot be compromised.</p>
                    </div>
                </div>

                <div className="insight-card">
                    <div className="insight-icon-wrapper orange">
                        <Cpu size={20} />
                    </div>
                    <div className="insight-content">
                        <h4>Parameter Efficiency</h4>
                        <p><strong>KD V2</strong> achieves the largest parameter reduction among the structured models, significantly lowering computational overhead.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DeploymentInsights;
