import React from 'react';
import './MetricCard.css';

const MetricCard = ({ title, value, icon: Icon, description, highlight }) => {
    return (
        <div className="metric-card">
            <div className="metric-card-header">
                <h3 className="metric-title">{title}</h3>
                {Icon && <Icon className="metric-icon" size={20} />}
            </div>
            
            <div className="metric-value-container">
                <span className="metric-value">{value}</span>
                {highlight && (
                    <div className="metric-trend">
                        <span className="trend-dot"></span>
                        <span className="metric-highlight">{highlight}</span>
                    </div>
                )}
            </div>
            
            <div className="metric-footer">
                <p className="metric-description">{description}</p>
                {/* Subtle sparkline placeholder decoration */}
                <div className="metric-sparkline">
                    <svg viewBox="0 0 100 20" preserveAspectRatio="none">
                        <path d="M0,15 L20,10 L40,18 L60,5 L80,12 L100,2" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="2" />
                    </svg>
                </div>
            </div>
            <div className="metric-glow"></div>
        </div>
    );
};

export default MetricCard;
