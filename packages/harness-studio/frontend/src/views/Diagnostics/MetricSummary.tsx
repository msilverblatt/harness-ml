import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface MetricsResponse {
    metrics?: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    error?: string;
}

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

export function MetricSummary({ runId }: MetricSummaryProps) {
    const { data, loading, error } = useApi<MetricsResponse>(`/api/runs/${runId}/metrics`);

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

    return (
        <div className={styles.metricsGrid}>
            {Object.entries(metrics).map(([name, value]) => (
                <div key={name} className={styles.metricCard} title={name}>
                    <div className={styles.metricName}>{name}</div>
                    <div className={styles.metricValue}>
                        {formatValue(value)}
                    </div>
                </div>
            ))}
        </div>
    );
}
