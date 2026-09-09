const rawUrl = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').trim().replace(/\/+$/, '');
// Ensure API_BASE_URL points to domain root without trailing slash or /api
export const API_BASE_URL = rawUrl.endsWith('/api') ? rawUrl.slice(0, -4) : rawUrl;

export const FALLBACK_BENCHMARK_DATA = [
    {
        id: "baseline",
        name: "Baseline ResNet-18",
        accuracy: "77.31%",
        size: "85.70 MB",
        size_reduction: "0.00%",
        accuracy_drop: "0.00 pp",
        cpu_inference: "20.99 ms/image"
    },
    {
        id: "int8",
        name: "INT8 Quantization",
        accuracy: "76.95%",
        size: "10.83 MB",
        size_reduction: "87.36%",
        accuracy_drop: "0.36 pp",
        cpu_inference: "11.31 ms/image"
    },
    {
        id: "pruned",
        name: "Pruning + Fine-tuning",
        accuracy: "75.82%",
        size: "27.39 MB",
        size_reduction: "68.04%",
        accuracy_drop: "1.49 pp",
        cpu_inference: "17.62 ms/image"
    },
    {
        id: "kd",
        name: "Knowledge Distillation V2",
        accuracy: "74.88%",
        size: "10.82 MB",
        size_reduction: "87.37%",
        accuracy_drop: "2.43 pp",
        cpu_inference: "8.38 ms/image"
    }
];

export const fetchBenchmarkData = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/benchmark`);
        if (!response.ok) {
            throw new Error(`Network response was not ok (${response.status})`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching benchmark data:', error);
        throw error;
    }
};

export const fetchModels = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching models:', error);
        throw error;
    }
};

export const fetchModelData = async (modelName) => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model/${encodeURIComponent(modelName)}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching model data for ${modelName}:`, error);
        throw error;
    }
};
