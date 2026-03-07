import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface MetricsResponse {
    metrics?: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    error?: string;
}

interface MetaCoefficientsProps {
    runId: string;
}

export function MetaCoefficients({ runId }: MetaCoefficientsProps) {
    const { data, loading, error } = useApi<MetricsResponse>(`/api/runs/${runId}/metrics`);

    if (loading) {
        return <div className={styles.emptyState}>Loading...</div>;
    }

    if (error) {
        return <div className={styles.emptyState}>Error: {error}</div>;
    }

    if (!data || data.error || !data.meta_coefficients || Object.keys(data.meta_coefficients).length === 0) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>No meta-learner coefficients available.</div>
            </div>
        );
    }

    const coeffs = data.meta_coefficients;
    const entries = Object.entries(coeffs).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));
    const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);

    return (
        <div className={styles.chartContainer}>
            <div className={styles.categoryHeader}>
                Meta-Learner Coefficients
            </div>
            <div className={styles.coeffSubtitle}>
                Higher absolute value = more influence in ensemble
            </div>
            <div className={styles.coeffList}>
                {entries.map(([name, value]) => {
                    const pct = (Math.abs(value) / maxAbs) * 100;
                    const isPositive = value >= 0;
                    const barColor = isPositive ? 'var(--color-success)' : 'var(--color-error)';
                    return (
                        <div key={name} className={styles.coeffRow}>
                            <span className={styles.coeffName}>{name}</span>
                            <div className={styles.coeffBarTrack}>
                                <div
                                    className={styles.coeffBarFill}
                                    style={{
                                        width: `${pct}%`,
                                        backgroundColor: barColor,
                                    }}
                                />
                            </div>
                            <span
                                className={styles.coeffValue}
                                style={{ color: barColor }}
                            >
                                {value >= 0 ? '+' : ''}{value.toFixed(3)}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
