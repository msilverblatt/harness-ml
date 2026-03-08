import { useState, useEffect, useRef } from 'react';

export interface Event {
    id: number;
    timestamp: string;
    tool: string;
    action: string;
    params: Record<string, unknown>;
    result: string;
    duration_ms: number;
    status: string;
    caller?: string;
}

interface UseWebSocketResult {
    events: Event[];
    connected: boolean;
}

const MAX_BACKOFF_MS = 30000;

function getWsUrl(path: string): string {
    if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_WS_URL) {
        return `${import.meta.env.VITE_WS_URL}${path}`;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${path}`;
}

export function useWebSocket(path: string = '/ws/events'): UseWebSocketResult {
    const [events, setEvents] = useState<Event[]>([]);
    const [connected, setConnected] = useState(false);
    const backoffRef = useRef(1000);
    const wsRef = useRef<WebSocket | null>(null);
    const mountedRef = useRef(true);

    useEffect(() => {
        mountedRef.current = true;
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

        function connect() {
            if (!mountedRef.current) return;

            const url = getWsUrl(path);
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                if (!mountedRef.current) return;
                setConnected(true);
                backoffRef.current = 1000;
            };

            ws.onmessage = (msg) => {
                if (!mountedRef.current) return;
                try {
                    const event: Event = JSON.parse(msg.data);
                    setEvents(prev => [...prev, event]);
                } catch {
                    // Ignore malformed messages
                }
            };

            ws.onclose = () => {
                if (!mountedRef.current) return;
                setConnected(false);
                reconnectTimer = setTimeout(() => {
                    backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
                    connect();
                }, backoffRef.current);
            };

            ws.onerror = () => {
                ws.close();
            };
        }

        connect();

        return () => {
            mountedRef.current = false;
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
            }
        };
    }, [path]);

    return { events, connected };
}
