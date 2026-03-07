import { useMemo } from 'react';
import type { Event } from './useWebSocket';

/**
 * Derive a refresh key from WebSocket events that changes whenever
 * a tool call completes (success or error). Pass to useApi's refreshKey
 * param to trigger live re-fetches.
 *
 * Optionally filter to specific tools so only relevant completions trigger refetch.
 */
export function useRefreshKey(events: Event[], tools?: string[]): number {
    return useMemo(() => {
        let count = 0;
        for (const e of events) {
            if (e.status !== 'success' && e.status !== 'error') continue;
            if (tools && !tools.includes(e.tool)) continue;
            count++;
        }
        return count;
    }, [events, tools]);
}
