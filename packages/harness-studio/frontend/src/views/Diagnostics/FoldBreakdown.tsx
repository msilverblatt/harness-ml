import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface FoldBreakdownProps {
    runId: string;
}

interface FoldData {
    fold: number;
    n_samples: number;
    [key: string]: unknown;
}

interface FoldsResponse {
    folds?: FoldData[];
    metric_names?: string[];
    error?: string;
}

function formatCell(value: unknown): string {
    if (typeof value === 'number') {
        if (Number.isInteger(value)) return String(value);
        if (Math.abs(value) >= 100) return value.toFixed(1);
        return value.toFixed(4);
    }
    return String(value ?? '');
}

export function FoldBreakdown({ runId }: FoldBreakdownProps) {
    const { data, loading, error } = useApi<FoldsResponse>(`/api/runs/${runId}/folds`);

    if (loading) {
        return <div className={styles.emptyState}>Loading fold breakdown...</div>;
    }

    if (error) {
        return <div className={styles.emptyState}>Error: {error}</div>;
    }

    if (!data || data.error || !data.folds || data.folds.length === 0) {
        return <div className={styles.emptyState}>No fold data available.</div>;
    }

    const { folds, metric_names } = data;
    const cols = metric_names ?? [];

    return (
        <div className={styles.chartContainer}>
            <table className={styles.foldTable}>
                <thead>
                    <tr>
                        <th>Fold</th>
                        <th>Samples</th>
                        {cols.map(col => (
                            <th key={col}>{col}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {folds.map((fold) => (
                        <tr key={fold.fold}>
                            <td>{fold.fold}</td>
                            <td>{fold.n_samples}</td>
                            {cols.map(col => (
                                <td key={col}>{formatCell(fold[col])}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
