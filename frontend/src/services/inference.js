const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const runInference = async (file, modelId) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);

    const response = await fetch(`${API_BASE_URL}/api/inference`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.statusText}`);
    }

    return await response.json();
};
