import { useState, useCallback } from 'react';
import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import { MetricLabel } from '../../components/Tooltip/Tooltip';
import styles from './Diagnostics.module.css';

interface ClassBalance {
    positive_rate: number;
    n_positive: number;
    n_total: number;
}

interface MetricsResponse {
    metrics?: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    class_balance?: ClassBalance;
    error?: string;
}

const IMBALANCE_THRESHOLD = 0.05;
const IMBALANCE_HIGHLIGHT_METRICS = new Set(['auc_pr', 'f1', 'precision', 'recall']);

interface MetricSummaryProps {
    runId: string;
}

function formatValue(value: unknown): string {
    if (typeof value === 'number') {
        if (Math.abs(value) >= 100) return value.toFixed(1);
        if (Math.abs(value) >= 1) return value.toFixed(4);
        return value.toFixed(6);
    }
    return String(value);
}

function metricsToMarkdown(metrics: Record<string, number>): string {
    const header = '| Metric | Value |';
    const divider = '| --- | --- |';
    const rows = Object.entries(metrics).map(
        ([name, value]) => `| ${name} | ${formatValue(value)} |`
    );
    return [header, divider, ...rows].join('\n');
}

function CopyButton({ text }: { text: string }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = useCallback(async () => {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch {
            // Fallback for non-HTTPS contexts
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        }
    }, [text]);

    return (
        <button
            className={styles.copyBtn}
            onClick={(e) => { e.stopPropagation(); handleCopy(); }}
            title="Copy to clipboard"
        >
            {copied ? 'Copied!' : 'Copy'}
        </button>
    );
}

export function MetricSummary({ runId }: MetricSummaryProps) {
    const project = useProject();
    const { data, loading, error } = useApi<MetricsResponse>(`/api/runs/${runId}/metrics`, undefined, project);

    if (loading) {
        return <div className={styles.emptyState}>Loading metrics...</div>;
    }

    if (error) {
        return <div className={styles.emptyState}>Error: {error}</div>;
    }

    if (!data || data.error || !data.metrics || Object.keys(data.metrics).length === 0) {
        return <div className={styles.emptyState}>No metrics available for this run.</div>;
    }

    const metrics = data.metrics;
    const count = Object.keys(metrics).length;
    const classBalance = data.class_balance;
    const isImbalanced = classBalance != null && classBalance.positive_rate < IMBALANCE_THRESHOLD;

    return (
        <div className={styles.metricsSection}>
            {isImbalanced && (
                <div className={styles.imbalanceBanner}>
                    Highly imbalanced ({(classBalance.positive_rate * 100).toFixed(1)}% positive). Focus on AUC-PR, F1, Precision, Recall.
                </div>
            )}
            <div className={styles.metricsSectionHeader}>
                {classBalance != null && (
                    <span className={styles.positiveRateBadge}>
                        {(classBalance.positive_rate * 100).toFixed(1)}% positive
                    </span>
                )}
                <CopyButton text={metricsToMarkdown(metrics)} />
            </div>
            <div
                className={styles.metricsGrid}
                style={{ '--metric-count': count } as React.CSSProperties}
            >
                {Object.entries(metrics).map(([name, value]) => {
                    const highlight = isImbalanced && IMBALANCE_HIGHLIGHT_METRICS.has(name.toLowerCase());
                    return (
                        <div
                            key={name}
                            className={`${styles.metricCard}${highlight ? ` ${styles.metricCardAccent}` : ''}`}
                            title={`Click to copy ${name}: ${formatValue(value)}`}
                        >
                            <span className={styles.metricValue}>
                                {formatValue(value)}
                            </span>
                            <span className={styles.metricName}><MetricLabel name={name} /></span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
