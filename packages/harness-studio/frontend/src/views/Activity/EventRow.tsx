import { useState, useEffect } from 'react';
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

function formatElapsed(timestamp: string): string {
    const diffMs = Date.now() - new Date(timestamp).getTime();
    if (diffMs < 0) return 'running 0s';
    const totalSeconds = Math.floor(diffMs / 1000);
    if (totalSeconds < 60) return `running ${totalSeconds}s`;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `running ${minutes}m ${seconds}s`;
}

function useElapsedTime(timestamp: string, active: boolean): string {
    const [elapsed, setElapsed] = useState(() => formatElapsed(timestamp));

    useEffect(() => {
        if (!active) return;
        setElapsed(formatElapsed(timestamp));
        const interval = setInterval(() => {
            setElapsed(formatElapsed(timestamp));
        }, 1000);
        return () => clearInterval(interval);
    }, [timestamp, active]);

    return elapsed;
}

function summarizeParams(params: Record<string, unknown>): string {
    const entries = Object.entries(params).filter(
        ([k]) => !['project_dir', 'ctx', 'current', 'total'].includes(k)
    );
    if (entries.length === 0) return '';
    return entries
        .slice(0, 3)
        .map(([k, v]) => {
            const val = typeof v === 'string' ? v : JSON.stringify(v);
            const truncated = val && val.length > 40 ? val.slice(0, 40) + '...' : val;
            return `${k}=${truncated}`;
        })
        .join(' ');
}

export function EventRow({ event }: EventRowProps) {
    const isRunning = event.status === 'running';
    const isProgress = event.status === 'progress';
    const isError = event.status === 'error';

    const progressParams = event.params as { current?: number; total?: number };
    const hasProgressData = isRunning && event.result && progressParams.total;

    const elapsed = useElapsedTime(event.timestamp, isRunning);

    const rowClass = isRunning ? styles.runningEvent
        : isError ? styles.statusError
        : undefined;

    const progressPercent = hasProgressData && progressParams.total
        ? Math.round(((progressParams.current ?? 0) / progressParams.total) * 100)
        : 0;

    const summary = (
        <div className={styles.eventSummary}>
            <span className={styles.timestamp}>{formatRelativeTime(event.timestamp)}</span>
            {isRunning ? (
                <span className={styles.runningBadge}>{event.tool}</span>
            ) : (
                <span
                    className={styles.toolBadge}
                    style={{ backgroundColor: getToolColor(event.tool) }}
                >
                    {event.tool}
                </span>
            )}
            <span className={styles.action}>
                {event.action}
                {isRunning && !hasProgressData && ' ...'}
                {hasProgressData && ` (${progressParams.current}/${progressParams.total})`}
                {isProgress && (() => {
                    const p = event.params as { current?: number; total?: number };
                    return p.total ? ` (${p.current}/${p.total})` : '';
                })()}
            </span>
            {hasProgressData ? (
                <span className={styles.params}>{event.result}</span>
            ) : isProgress ? (
                <span className={styles.params}>{event.result}</span>
            ) : (
                <span className={styles.params}>{summarizeParams(event.params)}</span>
            )}
            {!isRunning && !isProgress && (
                <span className={styles.duration}>{event.duration_ms}ms</span>
            )}
            {isRunning && (
                <span className={styles.elapsedLive}>{elapsed}</span>
            )}
        </div>
    );

    if (isRunning || isProgress) {
        return (
            <div className={rowClass}>
                {summary}
                {hasProgressData && (
                    <>
                        <div className={styles.progressBar}>
                            <div
                                className={styles.progressFill}
                                style={{ width: `${progressPercent}%` }}
                            />
                        </div>
                        {event.result && (
                            <div className={styles.progressMessage}>{event.result}</div>
                        )}
                    </>
                )}
            </div>
        );
    }

    return (
        <div className={rowClass}>
            <ExpandableRow detail={<MarkdownRenderer content={event.result} />}>
                {summary}
            </ExpandableRow>
        </div>
    );
}
