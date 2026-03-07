import { ExpandableRow } from '../../components/ExpandableRow/ExpandableRow';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import type { Event } from '../../hooks/useWebSocket';
import styles from './Activity.module.css';

interface EventRowProps {
    event: Event;
}

const TOOL_COLOR_MAP: Record<string, string> = {
    models: 'var(--color-tool-models)',
    pipeline: 'var(--color-tool-pipeline)',
    data: 'var(--color-tool-data)',
    features: 'var(--color-tool-features)',
    experiments: 'var(--color-tool-experiments)',
    config: 'var(--color-tool-config)',
};

function getToolColor(tool: string): string {
    return TOOL_COLOR_MAP[tool] ?? 'var(--color-text-muted)';
}

function formatRelativeTime(timestamp: string): string {
    const now = Date.now();
    const then = new Date(timestamp).getTime();
    const diffMs = now - then;

    if (diffMs < 0 || diffMs < 10000) return 'just now';

    const seconds = Math.floor(diffMs / 1000);
    if (seconds < 60) return `${seconds}s ago`;

    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;

    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;

    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

function summarizeParams(params: Record<string, unknown>): string {
    const entries = Object.entries(params);
    if (entries.length === 0) return '';
    return entries
        .slice(0, 3)
        .map(([k, v]) => {
            const val = typeof v === 'string' ? v : JSON.stringify(v);
            return `${k}=${val}`;
        })
        .join(' ');
}

export function EventRow({ event }: EventRowProps) {
    const summary = (
        <div className={styles.eventSummary}>
            <span className={styles.timestamp}>{formatRelativeTime(event.timestamp)}</span>
            <span
                className={styles.toolBadge}
                style={{ backgroundColor: getToolColor(event.tool) }}
            >
                {event.tool}
            </span>
            <span className={styles.action}>{event.action}</span>
            <span className={styles.params}>{summarizeParams(event.params)}</span>
            <span className={styles.duration}>{event.duration_ms}ms</span>
        </div>
    );

    return (
        <ExpandableRow detail={<MarkdownRenderer content={event.result} />}>
            {summary}
        </ExpandableRow>
    );
}
