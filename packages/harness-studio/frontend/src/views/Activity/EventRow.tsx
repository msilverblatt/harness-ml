import { useState, useEffect, useRef } from 'react';
import { ExpandableRow } from '../../components/ExpandableRow/ExpandableRow';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import type { Event } from '../../hooks/useWebSocket';
import { humanizeTimestamp, formatTimestamp } from '../../utils/time';
import styles from './Activity.module.css';

interface EventRowProps {
    event: Event;
    defaultExpanded?: boolean;
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
    return humanizeTimestamp(timestamp);
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

const SKIP_PARAMS = ['project_dir', 'ctx', 'current', 'total', 'action', 'experiment_ids',
    'overlay', 'primary_metric', 'variant', 'search_space', 'trial', 'detail', 'last_n'];

function getInlineSummary(event: Event): string {
    const p = event.params;

    // Experiment log_result: show experiment_id + verdict
    if (event.tool === 'experiments' && event.action === 'log_result') {
        const parts: string[] = [];
        if (p.experiment_id) parts.push(String(p.experiment_id));
        if (p.verdict) parts.push(String(p.verdict));
        if (p.description) {
            const desc = String(p.description);
            parts.push(desc.length > 60 ? desc.slice(0, 60) + '...' : desc);
        }
        return parts.join(' — ');
    }

    // Experiment quick_run: show experiment_id + description
    if (event.tool === 'experiments' && event.action === 'quick_run') {
        const parts: string[] = [];
        if (p.experiment_id) parts.push(String(p.experiment_id));
        if (p.description) {
            const desc = String(p.description);
            parts.push(desc.length > 80 ? desc.slice(0, 80) + '...' : desc);
        }
        return parts.join(' — ');
    }

    // Default: show key params
    const entries = Object.entries(p).filter(([k]) => !SKIP_PARAMS.includes(k));
    if (entries.length === 0) return '';
    return entries
        .slice(0, 3)
        .map(([k, v]) => {
            const val = typeof v === 'string' ? v : JSON.stringify(v);
            const truncated = val && val.length > 60 ? val.slice(0, 60) + '...' : val;
            return `${k}=${truncated}`;
        })
        .join('  ');
}

function getExpandedDetail(event: Event): string {
    const p = event.params;

    // Experiment log_result: build useful detail from params since result is just "Logged to journal"
    if (event.tool === 'experiments' && event.action === 'log_result') {
        const lines: string[] = [];
        if (p.experiment_id) lines.push(`## ${p.experiment_id}`);
        if (p.description) lines.push(`**${p.description}**`);
        if (p.hypothesis) lines.push(`\n**Hypothesis:** ${p.hypothesis}`);
        if (p.verdict) {
            const v = String(p.verdict);
            const icon = v === 'improved' ? '+' : v === 'regressed' ? '-' : '~';
            lines.push(`\n**Verdict:** ${icon} ${v}`);
        }
        if (p.conclusion) lines.push(`\n**Conclusion:** ${p.conclusion}`);
        return lines.join('\n');
    }

    // Default: use result field (which is markdown for quick_run, etc.)
    return event.result;
}

function FoldProgressBar({ current, total, message, phases }: {
    current: number;
    total: number;
    message: string;
    phases: number;
}) {
    // Parse model progress from message like "training xgb_main (2/3)"
    const modelMatch = message.match(/\((\d+)\/(\d+)\)/);
    const modelIdx = modelMatch ? parseInt(modelMatch[1]) : 0;
    const modelTotal = modelMatch ? parseInt(modelMatch[2]) : 1;

    // Track phase transitions: when current drops below prev, we've entered a new phase.
    // Use a ref to accumulate phase count across the lifetime of this progress bar.
    const phaseRef = useRef(0);
    const prevCurrentRef = useRef(current);

    // Detect phase transition: previous was at/near end of a phase, and current reset to start
    if (current < prevCurrentRef.current && prevCurrentRef.current >= total - 1) {
        phaseRef.current = Math.min(phaseRef.current + 1, phases - 1);
    }
    prevCurrentRef.current = current;

    const phase = phaseRef.current;

    // Simple fill percentage: each phase is an equal slice of the bar
    const phaseWidth = 100 / phases;
    const foldProgress = total > 0 ? current / total : 0;
    // Within a fold, add sub-model progress
    const subFoldProgress = modelTotal > 1 ? modelIdx / (total * modelTotal) : 0;
    const fillPct = phase * phaseWidth + (foldProgress + subFoldProgress) * phaseWidth;

    // Phase labels
    const phaseLabels = phases === 2
        ? ['experiment', 'baseline']
        : undefined;

    return (
        <div className={styles.foldProgress}>
            <div className={styles.foldTrack}>
                {phases > 1 ? (<>
                    {/* Phase 1 fill (experiment) */}
                    <div
                        className={styles.foldFillDone}
                        style={{ width: `${Math.min(fillPct, phaseWidth)}%` }}
                    />
                    {/* Phase 2 fill (baseline) */}
                    {fillPct > phaseWidth && (
                        <div
                            className={styles.foldFillBaseline}
                            style={{ left: `${phaseWidth}%`, width: `${fillPct - phaseWidth}%` }}
                        />
                    )}
                </>) : (
                    <div className={styles.foldFillDone} style={{ width: `${fillPct}%` }} />
                )}
                {/* Fold dividers */}
                {total > 1 && Array.from({ length: phases * total - 1 }, (_, i) => {
                    const pct = ((i + 1) / (phases * total)) * 100;
                    return (
                        <div
                            key={`f${i}`}
                            className={styles.foldTick}
                            style={{ left: `${pct}%` }}
                        />
                    );
                })}
                {/* Phase divider */}
                {phases > 1 && (
                    <div
                        className={styles.phaseTick}
                        style={{ left: `${phaseWidth}%` }}
                    />
                )}
            </div>
            <div className={styles.foldMeta}>
                {message && <span className={styles.progressMessage}>{message}</span>}
                <span className={styles.foldCount}>
                    {phaseLabels && `${phaseLabels[phase] ?? `phase ${phase + 1}`} · `}
                    fold {current + 1}/{total}
                </span>
            </div>
            {phases > 1 && (
                <div className={styles.phaseLabels}>
                    <span className={styles.phaseLabelExp}>experiment</span>
                    <span className={styles.phaseLabelBase}>baseline</span>
                </div>
            )}
        </div>
    );
}

export function EventRow({ event, defaultExpanded }: EventRowProps) {
    const isRunning = event.status === 'running';
    const isProgress = event.status === 'progress';
    const isError = event.status === 'error';

    const progressParams = event.params as { current?: number; total?: number };
    const hasProgressData = isRunning && event.result && progressParams.total;

    const elapsed = useElapsedTime(event.timestamp, isRunning);

    const isSuccess = event.status === 'success';
    const rowClass = isRunning ? styles.runningEvent
        : isError ? styles.statusError
        : isSuccess ? styles.statusSuccess
        : styles.statusDefault;

    const borderColor = isRunning ? 'var(--color-accent)'
        : isError ? 'var(--color-error)'
        : isSuccess ? 'var(--color-success)'
        : getToolColor(event.tool);

    const foldsCurrent = progressParams.current ?? 0;
    const foldsTotal = progressParams.total ?? 0;

    const inlineSummary = getInlineSummary(event);

    const summary = (
        <div className={styles.eventSummary}>
            <span className={styles.timestamp} title={formatTimestamp(event.timestamp)}>{formatRelativeTime(event.timestamp)}</span>
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
            {event.caller && (
                <span className={styles.callerBadge}>{event.caller}</span>
            )}
            {hasProgressData ? (
                <span className={styles.params}>{event.result}</span>
            ) : isProgress ? (
                <span className={styles.params}>{event.result}</span>
            ) : (
                <span className={styles.params}>{inlineSummary}</span>
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
            <div className={rowClass} style={{ borderLeftColor: borderColor }}>
                {summary}
                {hasProgressData && (
                    <FoldProgressBar
                        current={foldsCurrent}
                        total={foldsTotal}
                        message={event.result}
                        phases={event.tool === 'experiments' ? 2 : 1}
                    />
                )}
            </div>
        );
    }

    const detail = getExpandedDetail(event);

    return (
        <div className={rowClass} style={{ borderLeftColor: borderColor }}>
            <ExpandableRow detail={<MarkdownRenderer content={detail} />} defaultOpen={defaultExpanded}>
                {summary}
            </ExpandableRow>
        </div>
    );
}
