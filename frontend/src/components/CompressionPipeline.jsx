import React from 'react';
import { ArrowDown } from 'lucide-react';
import './CompressionPipeline.css';

const CompressionPipeline = () => {
    return (
        <div className="pipeline-container">
            <div className="pipeline-header">
                <h3>Compression Pipeline</h3>
                <p>Standard ResNet-18 optimization workflow for edge deployment.</p>
                <p className="pipeline-note">This workflow compresses the AI model itself—not uploaded images, PDFs, or other files.</p>
            </div>
            
            <div className="pipeline-flow">
                <div className="pipeline-node primary">
                    Original ResNet-18
                </div>
                
                <div className="pipeline-arrow"><ArrowDown size={20}/></div>
                
                <div className="pipeline-node secondary">
                    Compression Strategy
                </div>
                
                <div className="pipeline-branches">
                    <div className="pipeline-branch">
                        <div className="branch-line"></div>
                        <div className="pipeline-node method">
                            <h4>Quantization</h4>
                            <p>FP32 → INT8</p>
                        </div>
                    </div>
                    <div className="pipeline-branch">
                        <div className="branch-line"></div>
                        <div className="pipeline-node method">
                            <h4>Pruning</h4>
                            <p>Removes less important structure, then fine-tunes to recover accuracy.</p>
                        </div>
                    </div>
                    <div className="pipeline-branch">
                        <div className="branch-line"></div>
                        <div className="pipeline-node method">
                            <h4>Knowledge Distillation</h4>
                            <p>Teacher → Lightweight Student</p>
                        </div>
                    </div>
                </div>
                
                <div className="pipeline-arrow"><ArrowDown size={20}/></div>
                
                <div className="pipeline-node primary">
                    Compressed Model
                </div>
                
                <div className="pipeline-arrow"><ArrowDown size={20}/></div>
                
                <div className="pipeline-node accent">
                    Benchmark & Edge Deployment Analysis
                </div>
            </div>
        </div>
    );
};

export default CompressionPipeline;
