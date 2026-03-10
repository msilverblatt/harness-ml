import { useState, useEffect } from 'react';
import { MetricLabel, Tooltip } from '../../components/Tooltip/Tooltip';
import { humanizeTimestamp, formatTimestamp } from '../../utils/time';
import { isImprovement } from '../../utils/metrics';
import styles from './Experiments.module.css';

export interface Experiment {
    experiment_id: string;
    timestamp?: string;
    description?: string;
    hypothesis?: string;
    conclusion?: string;
    verdict?: string;
    metrics?: Record<string, number>;
    metric_std?: Record<string, number>;
    baseline_metrics?: Record<string, number>;
    primary_delta?: number;
    primary_metric?: string;
    notes?: string;
    [key: string]: unknown;
}

type SortKey = 'experiment_id' | 'timestamp' | 'hypothesis' | 'verdict' | 'primary_delta' | `metric_${string}`;

interface ExperimentTableProps {
    experiments: Experiment[];
    selectedIds: string[];
    onToggleCompare: (id: string) => void;
    initialExpandedId?: string | null;
    allMetricKeys: string[];
}

interface ExperimentDetail {
    id: string;
    hypothesis?: string;
    conclusion?: string;
    overlay?: Record<string, unknown>;
    results?: Record<string, unknown>;
}

function formatDate(ts?: string): string {
    if (!ts) return '--';
    return humanizeTimestamp(ts);
}

function fullDate(ts?: string): string {
    if (!ts) return '';
    return formatTimestamp(ts);
}

function truncate(s: string | undefined, max: number): string {
    if (!s) return '--';
    return s.length > max ? s.slice(0, max) + '...' : s;
}

function normalizeVerdict(v: string | undefined): string {
    if (!v) return 'pending';
    const lower = v.toLowerCase();
    if (['keep', 'improved', 'positive'].includes(lower)) return 'positive';
    if (['revert', 'regressed', 'negative', 'failed'].includes(lower)) return 'negative';
    if (['partial', 'neutral', 'baseline'].includes(lower)) return 'neutral';
    return 'pending';
}

function verdictClass(verdict?: string): string {
    const normalized = normalizeVerdict(verdict);
    switch (normalized) {
        case 'positive': return styles.verdictKeep;
        case 'neutral': return styles.verdictPartial;
        case 'negative': return styles.verdictRevert;
        default: return '';
    }
}

function formatDelta(delta?: number, primaryMetric?: string): { text: string; className: string } {
    if (delta == null) return { text: '--', className: '' };
    const sign = delta > 0 ? '+' : '';
    const text = `${sign}${delta.toFixed(4)}`;
    if (delta === 0) return { text, className: '' };
    const improved = primaryMetric ? isImprovement(primaryMetric, delta) : delta > 0;
    const className = improved ? styles.deltaPositive : styles.deltaNegative;
    return { text, className };
}

function metricDeltaClass(metricName: string, delta: number): string {
    if (delta === 0) return '';
    return isImprovement(metricName, delta) ? styles.deltaPositive : styles.deltaNegative;
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

function DetailPanel({ experiment, colSpan }: { experiment: Experiment; colSpan: number }) {
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
            <td colSpan={colSpan} className={styles.detailCell}>
                <div className={styles.detailPanel}>
                    <div className={styles.detailGrid}>
                        <div className={styles.detailBlock}>
                            <div className={styles.detailLabel}>Hypothesis</div>
                            <div className={styles.detailText}>{experiment.hypothesis || '--'}</div>
                            {loading && <div className={styles.detailLoading}>Loading overlay...</div>}
                            {!loading && overlaySummary.length > 0 && (
                                <div className={styles.overlayInline}>
                                    <div className={styles.detailLabel}>What Changed</div>
                                    <pre className={styles.overlayCode}>{overlaySummary.join('\n')}</pre>
                                </div>
                            )}
                            {!loading && overlay && overlaySummary.length === 0 && (
                                <div className={styles.overlayInline}>
                                    <div className={styles.detailLabel}>Overlay</div>
                                    <pre className={styles.overlayCode}>{JSON.stringify(overlay, null, 2)}</pre>
                                </div>
                            )}
                        </div>
                        <div className={styles.detailBlock}>
                            <div className={styles.detailConclusionHeader}>
                                <div className={styles.detailLabel}>Conclusion</div>
                                <span className={`${styles.verdictBadge} ${verdictClass(experiment.verdict)}`}>
                                    {experiment.verdict ?? '--'}
                                </span>
                            </div>
                            <div className={styles.detailText}>{experiment.conclusion || '--'}</div>
                            {metricKeys.length > 0 && (
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
                                                    <td><MetricLabel name={key} /></td>
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
                            )}
                        </div>
                    </div>
                </div>
            </td>
        </tr>
    );
}

export function ExperimentTable({ experiments, selectedIds, onToggleCompare, initialExpandedId, allMetricKeys }: ExperimentTableProps) {
    const [sortKey, setSortKey] = useState<SortKey>('timestamp');
    const [sortAsc, setSortAsc] = useState(false);
    const [expandedId, setExpandedId] = useState<string | null>(initialExpandedId ?? null);

    const totalColumns = 7 + allMetricKeys.length;

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
            default: {
                if (key.startsWith('metric_')) {
                    const metricName = key.slice(7);
                    return exp.metrics?.[metricName] ?? 0;
                }
                return '';
            }
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
        <div style={{ overflowX: 'auto' }}>
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
                        {allMetricKeys.map(mk => (
                            <th key={mk} className={styles.th} onClick={() => handleSort(`metric_${mk}`)}>
                                <MetricLabel name={mk} />{sortArrow(`metric_${mk}`)}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {sorted.map(exp => {
                        const isExpanded = expandedId === exp.experiment_id;
                        const delta = formatDelta(exp.primary_delta, exp.primary_metric);
                        return (
                            <ExperimentRow
                                key={exp.experiment_id}
                                experiment={exp}
                                isExpanded={isExpanded}
                                isCompareSelected={selectedIds.includes(exp.experiment_id)}
                                delta={delta}
                                onRowClick={handleRowClick}
                                onToggleCompare={onToggleCompare}
                                allMetricKeys={allMetricKeys}
                                totalColumns={totalColumns}
                            />
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

function formatMetricCell(experiment: Experiment, metricName: string): { text: string; className: string } {
    const val = experiment.metrics?.[metricName];
    if (val == null) return { text: '--', className: '' };
    const base = experiment.baseline_metrics?.[metricName];
    if (base == null) return { text: val.toFixed(4), className: '' };
    const delta = val - base;
    if (delta === 0) return { text: val.toFixed(4), className: '' };
    return { text: val.toFixed(4), className: isImprovement(metricName, delta) ? styles.deltaPositive : styles.deltaNegative };
}

function ExperimentRow({
    experiment,
    isExpanded,
    isCompareSelected,
    delta,
    onRowClick,
    onToggleCompare,
    allMetricKeys,
    totalColumns,
}: {
    experiment: Experiment;
    isExpanded: boolean;
    isCompareSelected: boolean;
    delta: { text: string; className: string };
    onRowClick: (id: string) => void;
    onToggleCompare: (id: string) => void;
    allMetricKeys: string[];
    totalColumns: number;
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
                <td className={styles.td} title={fullDate(experiment.timestamp)}>{formatDate(experiment.timestamp)}</td>
                <td className={styles.td} title={experiment.hypothesis}>{truncate(experiment.hypothesis, 50)}</td>
                <td className={styles.td}>
                    <span className={verdictClass(experiment.verdict)}>
                        {experiment.verdict ? <Tooltip term={experiment.verdict} label={experiment.verdict} /> : '--'}
                    </span>
                </td>
                <td className={`${styles.td} ${delta.className}`}>{delta.text}</td>
                {allMetricKeys.map(mk => {
                    const cell = formatMetricCell(experiment, mk);
                    return <td key={mk} className={`${styles.td} ${cell.className}`}>{cell.text}</td>;
                })}
            </tr>
            {isExpanded && <DetailPanel experiment={experiment} colSpan={totalColumns} />}
        </>
    );
}
