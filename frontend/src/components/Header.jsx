import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import './Header.css';

const Header = () => {
    return (
        <header className="header">
            <div className="header-titles">
                <h1>Edge Deployment Benchmark</h1>
                <p>Model Compression Evaluation • CIFAR-100 • ResNet-18</p>
            </div>
            <div className="header-actions">
                <div className="badge">
                    <span>Dataset:</span>
                    <strong>CIFAR-100</strong>
                </div>
                <div className="badge">
                    <span>Arch:</span>
                    <strong>ResNet-18</strong>
                </div>
                <div className="status-indicator">
                    <CheckCircle2 size={16} className="text-success" />
                    <span>Benchmark Complete</span>
                </div>
            </div>
        </header>
    );
};

export default Header;
