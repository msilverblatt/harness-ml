import { useState, useMemo, useRef, useEffect } from 'react';
import { useApi } from '../../hooks/useApi';
import type { Event } from '../../hooks/useWebSocket';
import { EventRow } from './EventRow';
import styles from './Activity.module.css';

interface EventLogProps {
    wsEvents: Event[];
}

export function EventLog({ wsEvents }: EventLogProps) {
    const { data: historicalEvents } = useApi<Event[]>('/api/events');
    const [toolFilter, setToolFilter] = useState('');
    const [textFilter, setTextFilter] = useState('');
    const prevCountRef = useRef(0);

    const allEvents = useMemo(() => {
        const historical = historicalEvents ?? [];
        const wsIds = new Set(wsEvents.map(e => e.id));
        const deduped = historical.filter(e => !wsIds.has(e.id));
        const combined = [...wsEvents, ...deduped];
        combined.sort((a, b) => {
            const ta = new Date(a.timestamp).getTime();
            const tb = new Date(b.timestamp).getTime();
            if (tb !== ta) return tb - ta;
            return b.id - a.id;
        });
        return combined;
    }, [historicalEvents, wsEvents]);

    const tools = useMemo(() => {
        const set = new Set(allEvents.map(e => e.tool));
        return Array.from(set).sort();
    }, [allEvents]);

    const filtered = useMemo(() => {
        return allEvents.filter(event => {
            if (toolFilter && event.tool !== toolFilter) return false;
            if (textFilter) {
                const search = textFilter.toLowerCase();
                const haystack = `${event.tool} ${event.action} ${event.result} ${JSON.stringify(event.params)}`.toLowerCase();
                if (!haystack.includes(search)) return false;
            }
            return true;
        });
    }, [allEvents, toolFilter, textFilter]);

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
                <input
                    className={styles.filterInput}
                    type="text"
                    placeholder="Search events..."
                    value={textFilter}
                    onChange={e => setTextFilter(e.target.value)}
                />
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
