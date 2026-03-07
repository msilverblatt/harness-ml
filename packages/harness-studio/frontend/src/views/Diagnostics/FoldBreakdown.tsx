import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface FoldBreakdownProps {
    runId: string;
}

export function FoldBreakdown({ runId }: FoldBreakdownProps) {
    const { data: metrics, loading, error } = useApi<Record<string, number>>(`/api/runs/${runId}/metrics`);

    if (loading) {
        return <div className={styles.emptyState}>Loading fold breakdown...</div>;
    }

    if (error) {
        return <div className={styles.emptyState}>Error: {error}</div>;
    }

    if (!metrics || Object.keys(metrics).length === 0) {
        return <div className={styles.emptyState}>No metrics available.</div>;
    }

    const entries = Object.entries(metrics).sort(([a], [b]) => a.localeCompare(b));

    return (
        <div>
            <div className={styles.categoryHeader}>Fold Breakdown</div>
            <div className={styles.emptyState} style={{ paddingBottom: 'var(--space-2)' }}>
                Per-fold breakdown available after backtest run
            </div>
            <div className={styles.chartContainer}>
                <table className={styles.foldTable}>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Pooled Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {entries.map(([name, value]) => (
                            <tr key={name}>
                                <td>{name}</td>
                                <td>{value.toFixed(4)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
