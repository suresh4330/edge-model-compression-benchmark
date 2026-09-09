const API_BASE_URL = 'http://127.0.0.1:8000/api';

export const fetchBenchmarkData = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/benchmark`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching benchmark data:', error);
        throw error;
    }
};

export const fetchModels = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
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
        const response = await fetch(`${API_BASE_URL}/model/${encodeURIComponent(modelName)}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching model data for ${modelName}:`, error);
        throw error;
    }
};
