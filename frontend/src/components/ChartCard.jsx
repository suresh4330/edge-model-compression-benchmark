import React from 'react';
import './ChartCard.css';

const ChartCard = ({ title, children, description }) => {
    return (
        <div className="chart-card">
            <div className="chart-card-header">
                <h3>{title}</h3>
                {description && <p>{description}</p>}
            </div>
            <div className="chart-card-content">
                {children}
            </div>
        </div>
    );
};

export default ChartCard;
