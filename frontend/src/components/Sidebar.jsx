import React from 'react';
import { LayoutDashboard, BarChart2, Cpu, Settings, Layers, Box, PlaySquare } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ activeSection, onNavigate }) => {
    const items = [
        { id: 'overview', label: 'OVERVIEW', icon: LayoutDashboard },
        { id: 'benchmark', label: 'BENCHMARK', icon: BarChart2 },
        { id: 'models', label: 'MODELS', icon: Box },
        { id: 'performance', label: 'PERFORMANCE', icon: Cpu },
        { id: 'compression', label: 'COMPRESSION', icon: Layers },
        { id: 'deployment', label: 'EDGE DEPLOYMENT', icon: Settings },
        { id: 'inference', label: 'INFERENCE LAB', icon: PlaySquare }
    ];

    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <h2>EDGE-BENCH</h2>
                <span>Model Compression Lab</span>
            </div>
            
            <nav className="sidebar-nav">
                {items.map(({ id, label, icon: Icon }) => (
                    <a
                        key={id}
                        href={`#${id}`}
                        onClick={(event) => onNavigate(event, id)}
                        className={`nav-item ${activeSection === id ? 'active' : ''}`}
                    >
                        <Icon size={20} />
                        <span>{label}</span>
                    </a>
                ))}
            </nav>

            <div className="sidebar-footer">
                <div className="footer-tag">
                    <span className="tag-label">Dataset</span>
                    <span className="tag-value">CIFAR-100</span>
                </div>
                <div className="footer-tag">
                    <span className="tag-label">Architecture</span>
                    <span className="tag-value">ResNet-18</span>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
