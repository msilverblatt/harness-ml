const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

export const api = {
  health: () => get<{ status: string }>('/health'),
  versions: {
    tree: () => get<{ current: string; versions: any[] }>('/versions/tree'),
    detail: (id: string) => get<any>(`/versions/${id}`),
    ancestry: (id: string) => get<any[]>(`/versions/${id}/ancestry`),
    compare: (v1: string, v2: string) => get<any>(`/versions/compare/${v1}/${v2}`),
  },
  pipeline: {
    config: () => get<any>('/pipeline/config'),
    dag: () => get<any>('/pipeline/dag'),
  },
  diagnostics: (id: string) => get<any>(`/diagnostics/${id}`),
  predictions: (id: string, limit = 50, offset = 0) =>
    get<any>(`/predictions/${id}?limit=${limit}&offset=${offset}`),
  data: {
    schema: () => get<any>('/data/schema'),
    profile: () => get<any>('/data/profile'),
  },
  monitor: {
    events: (limit = 50) => get<any[]>(`/monitor/events?limit=${limit}`),
    stats: () => get<any>('/monitor/stats'),
  },
};
