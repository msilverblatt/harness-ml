import { useState, useEffect } from 'react';
import styles from './Experiments.module.css';

export interface Experiment {
    experiment_id: string;
    timestamp?: string;
    description?: string;
    hypothesis?: string;
    conclusion?: string;
    verdict?: string;
    metrics?: Record<string, number>;
    baseline_metrics?: Record<string, number>;
    primary_delta?: number;
    primary_metric?: string;
    notes?: string;
    [key: string]: unknown;
}

type SortKey = 'experiment_id' | 'timestamp' | 'hypothesis' | 'verdict' | 'primary_delta';

interface ExperimentTableProps {
    experiments: Experiment[];
    selectedIds: string[];
    onToggleCompare: (id: string) => void;
}

interface ExperimentDetail {
    id: string;
    hypothesis?: string;
    conclusion?: string;
    overlay?: Record<string, unknown>;
    results?: Record<string, unknown>;
}

const LOWER_IS_BETTER = new Set(['brier', 'ece', 'log_loss', 'mae', 'mse', 'rmse']);

function formatDate(ts?: string): string {
    if (!ts) return '--';
    try {
        const d = new Date(ts);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
        return ts;
    }
}

function truncate(s: string | undefined, max: number): string {
    if (!s) return '--';
    return s.length > max ? s.slice(0, max) + '...' : s;
}

function verdictClass(verdict?: string): string {
    switch (verdict) {
        case 'keep':
        case 'improved': return styles.verdictKeep;
        case 'partial':
        case 'neutral': return styles.verdictPartial;
        case 'revert':
        case 'regressed': return styles.verdictRevert;
        default: return '';
    }
}

function formatDelta(delta?: number): { text: string; className: string } {
    if (delta == null) return { text: '--', className: '' };
    const sign = delta > 0 ? '+' : '';
    const text = `${sign}${delta.toFixed(4)}`;
    const className = delta > 0 ? styles.deltaPositive : delta < 0 ? styles.deltaNegative : '';
    return { text, className };
}

function metricDeltaClass(metricName: string, delta: number): string {
    if (delta === 0) return '';
    const lowerBetter = LOWER_IS_BETTER.has(metricName);
    const isImprovement = lowerBetter ? delta < 0 : delta > 0;
    return isImprovement ? styles.deltaPositive : styles.deltaNegative;
}

function getBaseUrl(): string {
    if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
        return import.meta.env.VITE_API_URL;
    }
    return window.location.origin;
}

function summarizeOverlay(overlay: Record<string, unknown>): string[] {
    const lines: string[] = [];
    for (const [section, value] of Object.entries(overlay)) {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
            const keys = Object.keys(value as Record<string, unknown>);
            if (keys.length > 0) {
                lines.push(`${section}: ${keys.join(', ')}`);
            }
        } else if (Array.isArray(value)) {
            lines.push(`${section}: [${value.length} items]`);
        } else {
            lines.push(`${section}: ${String(value)}`);
        }
    }
    return lines;
}

function DetailPanel({ experiment }: { experiment: Experiment }) {
    const [detail, setDetail] = useState<ExperimentDetail | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let cancelled = false;
        async function fetchDetail() {
            setLoading(true);
            try {
                const url = `${getBaseUrl()}/api/experiments/${experiment.experiment_id}`;
                const resp = await fetch(url);
                if (resp.ok) {
                    const json = await resp.json();
                    if (!cancelled) setDetail(json);
                }
            } catch {
                // silently ignore
            } finally {
                if (!cancelled) setLoading(false);
            }
        }
        fetchDetail();
        return () => { cancelled = true; };
    }, [experiment.experiment_id]);

    const allMetricKeys = new Set<string>();
    if (experiment.metrics) Object.keys(experiment.metrics).forEach(k => allMetricKeys.add(k));
    if (experiment.baseline_metrics) Object.keys(experiment.baseline_metrics).forEach(k => allMetricKeys.add(k));
    const metricKeys = Array.from(allMetricKeys).sort();

    const overlay = detail?.overlay;
    const overlaySummary = overlay ? summarizeOverlay(overlay) : [];

    return (
        <tr className={styles.detailRow}>
            <td colSpan={7} className={styles.detailCell}>
                <div className={styles.detailPanel}>
                    <div className={styles.detailGrid}>
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Hypothesis</div>
                            <div className={styles.detailText}>{experiment.hypothesis || '--'}</div>
                        </div>
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Conclusion</div>
                            <div className={styles.detailText}>{experiment.conclusion || '--'}</div>
                        </div>
                    </div>

                    <div className={styles.detailRow2}>
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Verdict</div>
                            <span className={`${styles.verdictBadge} ${verdictClass(experiment.verdict)}`}>
                                {experiment.verdict ?? '--'}
                            </span>
                        </div>
                    </div>

                    {metricKeys.length > 0 && (
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Metrics</div>
                            <table className={styles.detailMetricTable}>
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                        <th>Baseline</th>
                                        <th>Delta</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {metricKeys.map(key => {
                                        const val = experiment.metrics?.[key];
                                        const base = experiment.baseline_metrics?.[key];
                                        const delta = val != null && base != null ? val - base : undefined;
                                        const sign = delta != null && delta > 0 ? '+' : '';
                                        return (
                                            <tr key={key}>
                                                <td>{key}</td>
                                                <td>{val != null ? val.toFixed(4) : '--'}</td>
                                                <td>{base != null ? base.toFixed(4) : '--'}</td>
                                                <td className={delta != null ? metricDeltaClass(key, delta) : ''}>
                                                    {delta != null ? `${sign}${delta.toFixed(4)}` : '--'}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {loading && <div className={styles.detailLoading}>Loading overlay...</div>}

                    {!loading && overlaySummary.length > 0 && (
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Overlay Changes</div>
                            <pre className={styles.overlayCode}>{overlaySummary.join('\n')}</pre>
                        </div>
                    )}

                    {!loading && overlay && overlaySummary.length === 0 && (
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Overlay</div>
                            <pre className={styles.overlayCode}>{JSON.stringify(overlay, null, 2)}</pre>
                        </div>
                    )}
                </div>
            </td>
        </tr>
    );
}

export function ExperimentTable({ experiments, selectedIds, onToggleCompare }: ExperimentTableProps) {
    const [sortKey, setSortKey] = useState<SortKey>('timestamp');
    const [sortAsc, setSortAsc] = useState(false);
    const [expandedId, setExpandedId] = useState<string | null>(null);

    function handleSort(key: SortKey) {
        if (sortKey === key) {
            setSortAsc(!sortAsc);
        } else {
            setSortKey(key);
            setSortAsc(true);
        }
    }

    function handleRowClick(id: string) {
        setExpandedId(prev => prev === id ? null : id);
    }

    function getSortValue(exp: Experiment, key: SortKey): string | number {
        switch (key) {
            case 'experiment_id': return exp.experiment_id;
            case 'timestamp': return exp.timestamp ?? '';
            case 'hypothesis': return exp.hypothesis ?? '';
            case 'verdict': return exp.verdict ?? '';
            case 'primary_delta': return exp.primary_delta ?? 0;
        }
    }

    const sorted = [...experiments].sort((a, b) => {
        const aVal = getSortValue(a, sortKey);
        const bVal = getSortValue(b, sortKey);
        const cmp = typeof aVal === 'number' && typeof bVal === 'number'
            ? aVal - bVal
            : String(aVal).localeCompare(String(bVal));
        return sortAsc ? cmp : -cmp;
    });

    const sortArrow = (key: SortKey) =>
        sortKey === key ? <span className={styles.sortIndicator}>{sortAsc ? '\u25B2' : '\u25BC'}</span> : null;

    if (experiments.length === 0) {
        return <div className={styles.emptyState}>No experiments recorded yet.</div>;
    }

    return (
        <table className={styles.table}>
            <thead>
                <tr>
                    <th className={styles.th} style={{ width: 32 }}></th>
                    <th className={styles.th} style={{ width: 24 }}></th>
                    <th className={styles.th} onClick={() => handleSort('experiment_id')}>ID{sortArrow('experiment_id')}</th>
                    <th className={styles.th} onClick={() => handleSort('timestamp')}>Date{sortArrow('timestamp')}</th>
                    <th className={styles.th} onClick={() => handleSort('hypothesis')}>Hypothesis{sortArrow('hypothesis')}</th>
                    <th className={styles.th} onClick={() => handleSort('verdict')}>Verdict{sortArrow('verdict')}</th>
                    <th className={styles.th} onClick={() => handleSort('primary_delta')}>Delta{sortArrow('primary_delta')}</th>
                </tr>
            </thead>
            <tbody>
                {sorted.map(exp => {
                    const isExpanded = expandedId === exp.experiment_id;
                    const delta = formatDelta(exp.primary_delta);
                    return (
                        <ExperimentRow
                            key={exp.experiment_id}
                            experiment={exp}
                            isExpanded={isExpanded}
                            isCompareSelected={selectedIds.includes(exp.experiment_id)}
                            delta={delta}
                            onRowClick={handleRowClick}
                            onToggleCompare={onToggleCompare}
                        />
                    );
                })}
            </tbody>
        </table>
    );
}

function ExperimentRow({
    experiment,
    isExpanded,
    isCompareSelected,
    delta,
    onRowClick,
    onToggleCompare,
}: {
    experiment: Experiment;
    isExpanded: boolean;
    isCompareSelected: boolean;
    delta: { text: string; className: string };
    onRowClick: (id: string) => void;
    onToggleCompare: (id: string) => void;
}) {
    return (
        <>
            <tr
                className={`${styles.row} ${isExpanded ? styles.rowExpanded : ''}`}
                onClick={() => onRowClick(experiment.experiment_id)}
            >
                <td className={styles.td}>
                    <input
                        type="checkbox"
                        className={styles.checkbox}
                        checked={isCompareSelected}
                        onChange={(e) => {
                            e.stopPropagation();
                            onToggleCompare(experiment.experiment_id);
                        }}
                        onClick={(e) => e.stopPropagation()}
                    />
                </td>
                <td className={styles.td}>
                    <span className={`${styles.expandChevron} ${isExpanded ? styles.expandChevronOpen : ''}`}>
                        &#9656;
                    </span>
                </td>
                <td className={styles.td}>{experiment.experiment_id}</td>
                <td className={styles.td}>{formatDate(experiment.timestamp)}</td>
                <td className={styles.td} title={experiment.hypothesis}>{truncate(experiment.hypothesis, 50)}</td>
                <td className={styles.td}>
                    <span className={verdictClass(experiment.verdict)}>{experiment.verdict ?? '--'}</span>
                </td>
                <td className={`${styles.td} ${delta.className}`}>{delta.text}</td>
            </tr>
            {isExpanded && <DetailPanel experiment={experiment} />}
        </>
    );
}
