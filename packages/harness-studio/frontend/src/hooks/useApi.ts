import { useState, useEffect, useCallback } from 'react';

interface UseApiResult<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
    refetch: () => void;
}

function getBaseUrl(): string {
    if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
        return import.meta.env.VITE_API_URL;
    }
    return window.location.origin;
}

/**
 * Fetch data from an API endpoint.
 * Pass a `refreshKey` to trigger automatic refetches — when it changes, data is re-fetched.
 * Pass `project` to append ?project=X to the request URL.
 */
export function useApi<T>(path: string | null, refreshKey?: unknown, project?: string): UseApiResult<T> {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [tick, setTick] = useState(0);

    const refetch = useCallback(() => {
        setTick(prev => prev + 1);
    }, []);

    useEffect(() => {
        if (path === null) {
            setLoading(false);
            return;
        }

        let cancelled = false;
        const controller = new AbortController();

        async function fetchData() {
            setLoading(prev => data === null ? true : prev);
            setError(null);

            try {
                let url = `${getBaseUrl()}${path}`;
                if (project) {
                    const sep = url.includes('?') ? '&' : '?';
                    url += `${sep}project=${encodeURIComponent(project)}`;
                }
                const response = await fetch(url, { signal: controller.signal });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const json = await response.json();
                if (!cancelled) {
                    setData(json);
                }
            } catch (err) {
                if (!cancelled && !(err instanceof DOMException && err.name === 'AbortError')) {
                    setError(err instanceof Error ? err.message : String(err));
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        }

        fetchData();

        return () => {
            cancelled = true;
            controller.abort();
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [path, tick, refreshKey, project]);

    return { data, loading, error, refetch };
}
