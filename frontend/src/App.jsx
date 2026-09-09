import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, 
    ScatterChart, Scatter, ZAxis, ReferenceLine
} from 'recharts';
import { Target, HardDrive, Cpu, ArrowRight, Activity, Zap } from 'lucide-react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import MetricCard from './components/MetricCard';
import BenchmarkTable from './components/BenchmarkTable';
import ModelCard from './components/ModelCard';
import ChartCard from './components/ChartCard';
import CompressionPipeline from './components/CompressionPipeline';
import DeploymentInsights from './components/DeploymentInsights';
import RecommendationPanel from './components/RecommendationPanel';
import ModelDetail from './components/ModelDetail';
import InferenceLab from './components/InferenceLab';
import { fetchBenchmarkData, FALLBACK_BENCHMARK_DATA } from './services/api';
import './App.css';

const BENCHMARK_PARAMETERS = {
    baseline: '11,220,132',
    int8: 'N/A',
    pruned: '7,159,260',
    kd: '2,820,740'
};

const CustomParetoTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="custom-pareto-tooltip">
                <h4>{data.name}</h4>
                <div className="tooltip-metrics">
                    <div className="t-metric"><span className="t-label">Accuracy:</span> <span className="t-val">{data.accuracy}</span></div>
                    <div className="t-metric"><span className="t-label">Size:</span> <span className="t-val">{data.size}</span></div>
                    <div className="t-metric"><span className="t-label">Reduction:</span> <span className="t-val text-success">{data.size_reduction}</span></div>
                    <div className="t-metric"><span className="t-label">Latency:</span> <span className="t-val text-warning">{data.cpu_inference}</span></div>
                </div>
            </div>
        );
    }
    return null;
};

const App = () => {
    const [benchmarkData, setBenchmarkData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isUsingFallback, setIsUsingFallback] = useState(false);
    const [isReconnecting, setIsReconnecting] = useState(false);
    const [selectedModel, setSelectedModel] = useState(null);
    const [activeSection, setActiveSection] = useState('overview');

    const observer = useRef(null);

    const navigateToSection = (event, sectionId) => {
        event.preventDefault();
        const section = document.getElementById(sectionId);
        if (!section) return;

        setActiveSection(sectionId);
        window.history.pushState(null, '', `#${sectionId}`);
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    const loadData = async (isManualRetry = false) => {
        if (isManualRetry) setIsReconnecting(true);
        try {
            const data = await fetchBenchmarkData();
            setBenchmarkData(data.map((model) => ({
                ...model,
                // Existing recorded values from this repository's benchmark table.
                parameters: BENCHMARK_PARAMETERS[model.id] ?? 'N/A'
            })));
            setIsUsingFallback(false);
            setError(null);
            setLoading(false);
        } catch (err) {
            console.warn("Backend unavailable or spinning up; using recorded benchmark data:", err);
            setBenchmarkData(FALLBACK_BENCHMARK_DATA.map((model) => ({
                ...model,
                parameters: BENCHMARK_PARAMETERS[model.id] ?? 'N/A'
            })));
            setIsUsingFallback(true);
            setLoading(false);
        } finally {
            if (isManualRetry) setIsReconnecting(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    useEffect(() => {
        const handleIntersect = (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    setActiveSection(entry.target.id);
                    // Optionally update URL hash without scrolling
                    window.history.replaceState(null, null, `#${entry.target.id}`);
                }
            });
        };

        observer.current = new IntersectionObserver(handleIntersect, {
            root: null,
            rootMargin: '-50% 0px -50% 0px', // Trigger when section is in middle of screen
            threshold: 0
        });

        const sections = document.querySelectorAll('.dashboard-section');
        sections.forEach((section) => observer.current.observe(section));

        return () => {
            if (observer.current) {
                observer.current.disconnect();
            }
        };
    }, [loading]); // Re-bind observer after loading finishes

    useEffect(() => {
        const navigateFromHash = () => {
            const sectionId = window.location.hash.slice(1);
            const section = document.getElementById(sectionId);
            if (section) {
                setActiveSection(sectionId);
                section.scrollIntoView({ behavior: 'auto', block: 'start' });
            }
        };

        navigateFromHash();
        window.addEventListener('hashchange', navigateFromHash);
        return () => window.removeEventListener('hashchange', navigateFromHash);
    }, [loading]);

    const parsedData = useMemo(() => {
        return benchmarkData.map(model => ({
            ...model,
            accVal: parseFloat(model.accuracy),
            sizeVal: parseFloat(model.size),
            latVal: parseFloat(model.cpu_inference),
            redVal: parseFloat(model.size_reduction),
            shortName: model.name.replace(' + Fine-tuning', '').replace(' V2', '').replace('Baseline ', '')
        }));
    }, [benchmarkData]);

    const baselineModel = benchmarkData.find(m => m.name.includes('Baseline'));

    if (loading) {
        return <div className="loading-screen">
            <div className="loader"></div>
            <p>Initializing Research Dashboard...</p>
        </div>;
    }

    if (error) {
        return <div className="error-screen">
            <h2>Connection Error</h2>
            <p>{error}</p>
        </div>;
    }

    return (
        <div className="app-layout">
            <Sidebar activeSection={activeSection} onNavigate={navigateToSection} />
            <main className="main-content">
                <Header />
                
                {isUsingFallback && (
                    <div style={{
                        margin: '1.25rem 2rem 0',
                        padding: '0.85rem 1.25rem',
                        borderRadius: '10px',
                        backgroundColor: 'rgba(234, 179, 8, 0.1)',
                        border: '1px solid rgba(234, 179, 8, 0.3)',
                        color: '#fef08a',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '1rem',
                        fontSize: '0.875rem'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <Activity size={18} style={{ color: '#eab308', flexShrink: 0 }} />
                            <span>
                                <strong>Live API Notice:</strong> The backend is currently waking up (free-tier Render instances sleep when inactive). Displaying recorded benchmark metrics below.
                            </span>
                        </div>
                        <button
                            onClick={() => loadData(true)}
                            disabled={isReconnecting}
                            style={{
                                backgroundColor: '#eab308',
                                color: '#1a1a1a',
                                border: 'none',
                                borderRadius: '6px',
                                padding: '0.45rem 0.9rem',
                                cursor: isReconnecting ? 'not-allowed' : 'pointer',
                                fontWeight: '600',
                                fontSize: '0.8rem',
                                whiteSpace: 'nowrap'
                            }}
                        >
                            {isReconnecting ? 'Connecting...' : 'Retry Live API'}
                        </button>
                    </div>
                )}
                
                <div className="dashboard-container">
                    
                    {/* HERO OVERVIEW */}
                    <section id="overview" className="dashboard-section">
                        <div className="hero-grid">
                            <div className="hero-text-content">
                                <h2>Edge Deployment Model Compression</h2>
                                <p className="hero-subtitle">Benchmarking accuracy, model size, parameter efficiency, and CPU inference across compression techniques.</p>
                                <div className="hero-badges">
                                    <span className="tech-badge"><Activity size={14}/> PyTorch</span>
                                    <span className="tech-badge"><Cpu size={14}/> Edge CPU</span>
                                </div>
                            </div>
                            
                            <div className="hero-visual-card">
                                <h3>Compression Impact</h3>
                                <div className="visual-story">
                                    <div className="story-node baseline-node">
                                        <div className="node-title">Baseline ResNet-18</div>
                                        <div className="node-metrics">
                                            <span>85.70 MB</span>
                                            <span>77.31%</span>
                                            <span>20.99 ms</span>
                                        </div>
                                    </div>
                                    
                                    <div className="story-connector">
                                        <ArrowRight className="connector-arrow text-accent" />
                                        <div className="reduction-pill">↓ 87.37% Size</div>
                                    </div>
                                    
                                    <div className="story-node compressed-node">
                                        <div className="node-title text-success">Optimized Edge Model</div>
                                        <div className="node-metrics">
                                            <span className="text-success">10.82 MB</span>
                                            <span className="text-warning">74.88%</span>
                                            <span className="text-success">8.38 ms</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* EXECUTIVE METRICS */}
                        <div className="metrics-grid">
                            <MetricCard 
                                title="Baseline Accuracy" 
                                value="77.31%" 
                                icon={Target}
                                description="Reference Point"
                            />
                            <MetricCard 
                                title="Smallest Model" 
                                value="10.82 MB" 
                                icon={HardDrive}
                                description="Knowledge Distillation V2"
                                highlight="87.37% reduction"
                            />
                            <MetricCard 
                                title="Accuracy Retention" 
                                value="-0.36 pp" 
                                icon={Activity}
                                description="Best preserved (INT8)"
                            />
                            <MetricCard 
                                title="Fastest CPU Inference" 
                                value="8.38 ms" 
                                icon={Zap}
                                description="Per image latency"
                                highlight="2.5x speedup"
                            />
                        </div>
                    </section>

                    {/* MAIN BENCHMARK */}
                    <section id="benchmark" className="dashboard-section">
                        <h2 className="section-title">Detailed Results Table</h2>
                        <BenchmarkTable 
                            models={benchmarkData} 
                            onRowClick={setSelectedModel}
                        />
                        <div className="charts-grid-2 benchmark-charts">
                            <ChartCard title="Accuracy vs Compression Trade-off" description="Compare accuracy retention against model compactness.">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#2D333F" />
                                        <XAxis type="number" dataKey="redVal" name="Size Reduction" unit="%" domain={[-5, 95]} tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={{stroke: '#2D333F'}} tickLine={false} label={{ value: 'Size Reduction %', position: 'insideBottom', offset: -15, fill: '#6B7280', fontSize: 12 }} />
                                        <YAxis type="number" dataKey="accVal" name="Accuracy" unit="%" domain={[73, 79]} tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={{stroke: '#2D333F'}} tickLine={false} label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft', fill: '#6B7280', fontSize: 12 }} />
                                        <ZAxis type="category" dataKey="name" name="Model" />
                                        <RechartsTooltip content={<CustomParetoTooltip />} cursor={{strokeDasharray: '3 3', stroke: '#3B82F6'}} />
                                        <Scatter name="Models" data={parsedData} fill="#8B5CF6" />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </ChartCard>
                            <ChartCard title="Model Size Comparison" description="Storage footprint (MB)"><ResponsiveContainer width="100%" height="100%"><BarChart data={parsedData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2D333F" /><XAxis dataKey="shortName" tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} /><YAxis tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} /><RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.02)'}} contentStyle={{backgroundColor: '#1D2128', borderColor: '#2D333F'}} /><Bar dataKey="sizeVal" fill="#3B82F6" radius={[4, 4, 0, 0]} name="Size (MB)" /></BarChart></ResponsiveContainer></ChartCard>
                            <ChartCard title="Accuracy Comparison" description="Top-1 Accuracy (%)"><ResponsiveContainer width="100%" height="100%"><BarChart data={parsedData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2D333F" /><XAxis dataKey="shortName" tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} /><YAxis domain={[70, 80]} tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} /><RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.02)'}} contentStyle={{backgroundColor: '#1D2128', borderColor: '#2D333F'}} /><Bar dataKey="accVal" fill="#10B981" radius={[4, 4, 0, 0]} name="Accuracy (%)" /><ReferenceLine y={77.31} stroke="#2D333F" strokeDasharray="3 3" /></BarChart></ResponsiveContainer></ChartCard>
                            <ChartCard title="CPU Inference Latency" description="Benchmark CPU latency in milliseconds per image (ms/image)."><ResponsiveContainer width="100%" height="100%"><BarChart data={parsedData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2D333F" /><XAxis dataKey="shortName" tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} /><YAxis tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} /><RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.02)'}} contentStyle={{backgroundColor: '#1D2128', borderColor: '#2D333F'}} /><Bar dataKey="latVal" fill="#F59E0B" radius={[4, 4, 0, 0]} name="Latency (ms)" /></BarChart></ResponsiveContainer></ChartCard>
                        </div>
                    </section>

                    {/* MODELS EXPLORER */}
                    <section id="models" className="dashboard-section">
                        <div className="section-header">
                            <h2 className="section-title">Model Explorer</h2>
                        </div>
                        <div className="model-cards-grid">
                            {benchmarkData.map((model, idx) => (
                                <ModelCard 
                                    key={idx} 
                                    model={model} 
                                    onClick={setSelectedModel} 
                                />
                            ))}
                        </div>
                    </section>

                    {/* PERFORMANCE ANALYTICS */}
                    <section id="performance" className="dashboard-section">
                        <div className="section-header">
                            <div>
                                <h2 className="section-title">CPU Performance</h2>
                                <p className="section-subtitle">Benchmark latency below is a controlled CPU measurement. Live Inference Lab timing is a separate per-request measurement that also includes the current runtime environment.</p>
                            </div>
                        </div>
                        <div className="charts-grid-2">
                            <ChartCard title="CPU Inference Latency" description="Controlled benchmark measurement in milliseconds per image (ms/image).">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={parsedData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2D333F" />
                                        <XAxis dataKey="shortName" tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} />
                                        <YAxis tick={{fill: '#9CA3AF', fontSize: 12}} axisLine={false} tickLine={false} />
                                        <RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.02)'}} contentStyle={{backgroundColor: '#1D2128', borderColor: '#2D333F'}} />
                                        <Bar dataKey="latVal" fill="#F59E0B" radius={[4, 4, 0, 0]} name="Latency (ms)" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartCard>
                        </div>
                    </section>
                    
                    {/* KEY FINDINGS */}
                    <section className="dashboard-section key-findings">
                        <h2 className="section-title">Key Findings</h2>
                        <div className="findings-grid">
                            <div className="finding-card">
                                <h4>INT8 preserves accuracy best</h4>
                                <p>76.95% accuracy with only a 0.36 percentage point drop from baseline.</p>
                            </div>
                            <div className="finding-card">
                                <h4>KD V2 achieves fastest inference</h4>
                                <p>Drops CPU latency down to just 8.38 ms/image, ideal for real-time edge streaming.</p>
                            </div>
                            <div className="finding-card">
                                <h4>Maximum footprint reduction</h4>
                                <p>Both INT8 and KD V2 achieve approximately 87% model size reduction (~10.8 MB).</p>
                            </div>
                            <div className="finding-card">
                                <h4>Pruning is a middle-ground</h4>
                                <p>Provides a balanced compression strategy with 68% size reduction and moderate speedup.</p>
                            </div>
                        </div>
                    </section>

                    {/* COMPRESSION PIPELINE */}
                    <section id="compression" className="dashboard-section">
                        <h2 className="section-title">Methodology Workflow</h2>
                        <CompressionPipeline />
                    </section>

                    {/* EDGE DEPLOYMENT */}
                    <section id="deployment" className="dashboard-section">
                        <h2 className="section-title">Edge Deployment Strategy</h2>
                        <div className="charts-grid-2">
                            <RecommendationPanel />
                            <DeploymentInsights />
                        </div>
                    </section>

                    {/* INFERENCE LAB */}
                    <InferenceLab />

                    {/* ADDITIONAL EXPERIMENT */}
                    <section className="dashboard-section extra-experiment">
                        <h2 className="section-title">Additional Experiment</h2>
                        <div className="extra-card">
                            <div className="extra-header">
                                <h3>Improved Baseline ResNet-18</h3>
                                <span className="extra-badge">Reference Only</span>
                            </div>
                            <div className="extra-metrics">
                                <div className="extra-metric">
                                    <span>Accuracy:</span>
                                    <strong>78.22%</strong>
                                </div>
                                <div className="extra-metric">
                                    <span>Model Size:</span>
                                    <strong>85.70 MB</strong>
                                </div>
                            </div>
                            <p className="extra-desc">This experiment improves the baseline training result but is kept separate from the main compression benchmark.</p>
                        </div>
                    </section>
                    
                    {/* FOOTER */}
                    <footer className="dashboard-footer">
                        <div className="footer-content">
                            <div className="footer-brand">
                                <strong>EDGE-BENCH</strong>
                                <span>Edge Deployment Model Compression Benchmark</span>
                            </div>
                            <div className="footer-meta">
                                <span>CIFAR-100 • ResNet-18</span>
                                <span className="footer-status">Benchmark Status: <strong className="text-success">Complete</strong></span>
                            </div>
                        </div>
                    </footer>
                </div>
            </main>
            
            {/* MODEL DETAIL MODAL */}
            {selectedModel && (
                <ModelDetail 
                    model={selectedModel} 
                    baseline={baselineModel}
                    onClose={() => setSelectedModel(null)} 
                />
            )}
        </div>
    );
};

export default App;
