import { useState, useMemo, useRef, useEffect } from 'react';
import { useApi } from '../../hooks/useApi';
import type { Event } from '../../hooks/useWebSocket';
import { EventRow } from './EventRow';
import styles from './Activity.module.css';

interface EventLogProps {
    wsEvents: Event[];
}

/**
 * Merge running/progress/completed events into unified event groups.
 * A "running" event starts a group. Progress events update it. A success/error event closes it.
 * This prevents duplicate rows for start + finish of the same tool call.
 */
function mergeEvents(events: Event[]): Event[] {
    // Process chronologically (oldest first) to build groups
    const chronological = [...events].sort((a, b) => {
        const ta = new Date(a.timestamp).getTime();
        const tb = new Date(b.timestamp).getTime();
        if (ta !== tb) return ta - tb;
        return a.id - b.id;
    });

    // Track open "running" events by tool+action
    const openGroups = new Map<string, Event>();
    const merged: Event[] = [];
    const consumed = new Set<number>();

    for (const event of chronological) {
        const key = `${event.tool}:${event.action}`;

        if (event.status === 'running') {
            // Start a new group
            openGroups.set(key, { ...event });
            continue;
        }

        if (event.status === 'progress') {
            // Update the open group's display
            const group = openGroups.get(key);
            if (group) {
                // Update the group with latest progress info but keep 'running' status
                // so it continues to show the running style until success/error
                group.result = event.result;
                group.params = event.params;
                consumed.add(event.id);
                continue;
            }
            // No open group — show standalone
        }

        if (event.status === 'success' || event.status === 'error') {
            // Close the open group — replace with completion event
            if (openGroups.has(key)) {
                openGroups.delete(key);
                // The completed event supersedes the running one
            }
        }

        merged.push(event);
    }

    // Add still-open groups (currently running)
    for (const group of openGroups.values()) {
        merged.push(group);
    }

    // Sort newest first
    merged.sort((a, b) => {
        const ta = new Date(a.timestamp).getTime();
        const tb = new Date(b.timestamp).getTime();
        if (tb !== ta) return tb - ta;
        return b.id - a.id;
    });

    return merged;
}

export function EventLog({ wsEvents }: EventLogProps) {
    const { data: historicalEvents } = useApi<Event[]>('/api/events');
    const [toolFilter, setToolFilter] = useState('');
    const [textFilter, setTextFilter] = useState('');
    const [statusFilter, setStatusFilter] = useState('');
    const prevCountRef = useRef(0);

    const allEvents = useMemo(() => {
        const historical = historicalEvents ?? [];
        const wsIds = new Set(wsEvents.map(e => e.id));
        const deduped = historical.filter(e => !wsIds.has(e.id));
        const combined = [...wsEvents, ...deduped];
        return mergeEvents(combined);
    }, [historicalEvents, wsEvents]);

    const tools = useMemo(() => {
        const set = new Set(allEvents.map(e => e.tool));
        return Array.from(set).sort();
    }, [allEvents]);

    const filtered = useMemo(() => {
        return allEvents.filter(event => {
            if (toolFilter && event.tool !== toolFilter) return false;
            if (statusFilter && event.status !== statusFilter) return false;
            if (textFilter) {
                const search = textFilter.toLowerCase();
                const haystack = `${event.tool} ${event.action} ${event.result} ${JSON.stringify(event.params)}`.toLowerCase();
                if (!haystack.includes(search)) return false;
            }
            return true;
        });
    }, [allEvents, toolFilter, textFilter, statusFilter]);

    const hasActiveFilters = toolFilter !== '' || textFilter !== '' || statusFilter !== '';

    const clearFilters = () => {
        setToolFilter('');
        setTextFilter('');
        setStatusFilter('');
    };

    const newEventIds = useRef(new Set<number>());

    useEffect(() => {
        const currentCount = wsEvents.length;
        if (currentCount > prevCountRef.current) {
            const newEvents = wsEvents.slice(prevCountRef.current);
            for (const e of newEvents) {
                newEventIds.current.add(e.id);
            }
        }
        prevCountRef.current = currentCount;
    }, [wsEvents]);

    return (
        <div className={styles.eventLog}>
            <div className={styles.filterBar}>
                <select
                    className={styles.filterSelect}
                    value={toolFilter}
                    onChange={e => setToolFilter(e.target.value)}
                >
                    <option value="">All tools</option>
                    {tools.map(tool => (
                        <option key={tool} value={tool}>{tool}</option>
                    ))}
                </select>
                <select
                    className={styles.filterSelect}
                    value={statusFilter}
                    onChange={e => setStatusFilter(e.target.value)}
                >
                    <option value="">All statuses</option>
                    <option value="running">Running</option>
                    <option value="success">Success</option>
                    <option value="error">Error</option>
                </select>
                <input
                    className={styles.filterInput}
                    type="text"
                    placeholder="Search events..."
                    value={textFilter}
                    onChange={e => setTextFilter(e.target.value)}
                />
                <span className={styles.filterCount}>
                    {filtered.length} of {allEvents.length} events
                </span>
                {hasActiveFilters && (
                    <button className={styles.clearButton} onClick={clearFilters}>
                        Clear
                    </button>
                )}
            </div>
            <div className={styles.eventList}>
                {filtered.map(event => (
                    <div
                        key={event.id}
                        className={newEventIds.current.has(event.id) ? styles.newEvent : undefined}
                    >
                        <EventRow event={event} />
                    </div>
                ))}
            </div>
        </div>
    );
}
