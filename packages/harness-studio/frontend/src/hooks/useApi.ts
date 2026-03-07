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

export function useApi<T>(path: string): UseApiResult<T> {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [tick, setTick] = useState(0);

    const refetch = useCallback(() => {
        setTick(prev => prev + 1);
    }, []);

    useEffect(() => {
        let cancelled = false;
        const controller = new AbortController();

        async function fetchData() {
            setLoading(true);
            setError(null);

            try {
                const url = `${getBaseUrl()}${path}`;
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
    }, [path, tick]);

    return { data, loading, error, refetch };
}
