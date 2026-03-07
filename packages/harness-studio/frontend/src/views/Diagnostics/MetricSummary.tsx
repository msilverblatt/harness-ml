import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

const METRIC_CATEGORIES: Record<string, string[]> = {
    'Probability': ['brier', 'log_loss', 'brier_skill'],
    'Classification': ['accuracy', 'f1', 'mcc', 'kappa', 'precision', 'recall', 'balanced_accuracy'],
    'Calibration': ['ece', 'mce', 'calibration_slope', 'calibration_intercept'],
    'Ensemble': ['lomo_mean', 'diversity_score', 'correlation_mean'],
};

const KNOWN_METRICS = new Set(
    Object.values(METRIC_CATEGORIES).flat()
);

interface MetricSummaryProps {
    runId: string;
}

function formatValue(value: number): string {
    return value.toFixed(4);
}

export function MetricSummary({ runId }: MetricSummaryProps) {
    const { data: metrics, loading, error } = useApi<Record<string, number>>(`/api/runs/${runId}/metrics`);

    if (loading) {
        return <div className={styles.emptyState}>Loading metrics...</div>;
    }

    if (error) {
        return <div className={styles.emptyState}>Error: {error}</div>;
    }

    if (!metrics || Object.keys(metrics).length === 0) {
        return <div className={styles.emptyState}>No metrics available for this run.</div>;
    }

    const categorized = new Set<string>();
    const otherMetrics: string[] = [];

    for (const key of Object.keys(metrics)) {
        if (!KNOWN_METRICS.has(key)) {
            otherMetrics.push(key);
        }
    }

    const categories = { ...METRIC_CATEGORIES };
    if (otherMetrics.length > 0) {
        categories['Other'] = otherMetrics;
    }

    return (
        <div>
            {Object.entries(categories).map(([category, metricNames]) => {
                const available = metricNames.filter(name => name in metrics);
                if (available.length === 0) return null;

                available.forEach(name => categorized.add(name));

                return (
                    <div key={category}>
                        <div className={styles.categoryHeader}>{category}</div>
                        <div className={styles.metricsGrid}>
                            {available.map(name => (
                                <div key={name} className={styles.metricCard} title={name}>
                                    <div className={styles.metricName}>{name}</div>
                                    <div className={styles.metricValue}>
                                        {formatValue(metrics[name])}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
