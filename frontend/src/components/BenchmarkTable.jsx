import React from 'react';
import './BenchmarkTable.css';

const BenchmarkTable = ({ models, onRowClick }) => {
    return (
        <div className="benchmark-table-container">
            <table className="benchmark-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Model Size</th>
                        <th>Parameters</th>
                        <th>Size Reduction</th>
                        <th>Accuracy Drop</th>
                        <th>CPU Inference</th>
                    </tr>
                </thead>
                <tbody>
                    {models.map((model, index) => {
                        const isBaseline = model.name.includes('Baseline');
                        return (
                            <tr 
                                key={index} 
                                className={isBaseline ? 'row-baseline' : ''}
                                onClick={() => onRowClick(model)}
                            >
                                <td className="td-model-name">
                                    {model.name}
                                    {isBaseline && <span className="badge-ref">REF</span>}
                                </td>
                                <td>{model.accuracy}</td>
                                <td>{model.size}</td>
                                <td className="text-muted">{model.parameters}</td>
                                <td className={model.size_reduction !== "0.00%" ? "text-success" : "text-muted"}>
                                    {model.size_reduction}
                                </td>
                                <td className={model.accuracy_drop !== "0.00 pp" ? "text-warning" : "text-muted"}>
                                    {model.accuracy_drop}
                                </td>
                                <td>{model.cpu_inference}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};

export default BenchmarkTable;
