const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Request failed with ${response.status}: ${details}`);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }

  return response.text();
}

export async function checkBackendHealth() {
  return request('/health');
}

export async function healthCheck() {
  return checkBackendHealth();
}

export async function createInvestigation(payload) {
  return request('/investigations', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function submitInvestigation(payload) {
  return createInvestigation(payload);
}

export async function createInvestigationRequest(payload) {
  return createInvestigation(payload);
}