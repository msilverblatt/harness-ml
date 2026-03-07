import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface CorrelationResponse {
    models?: string[];
    matrix?: number[][];
    error?: string;
}

interface CorrelationMatrixProps {
    runId: string;
}

function getCorrelationColor(value: number): string {
    if (value >= 0.99) return 'var(--color-text-muted)';
    if (value > 0.8) return '#f8514966';
    if (value > 0.6) return '#d2992266';
    return '#3fb95066';
}

export function CorrelationMatrix({ runId }: CorrelationMatrixProps) {
    const { data, loading, error } = useApi<CorrelationResponse>(`/api/runs/${runId}/correlations`);

    if (loading) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>Loading correlations...</div>
            </div>
        );
    }

    if (error || !data || data.error || !data.models || data.models.length === 0) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>
                    {data?.error ?? error ?? 'No correlation data available.'}
                </div>
            </div>
        );
    }

    const { models, matrix } = data;
    if (!matrix) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>No correlation matrix available.</div>
            </div>
        );
    }

    return (
        <div className={styles.chartContainer}>
            <div className={styles.categoryHeader}>Prediction Correlations</div>
            <div
                className={styles.correlationGrid}
                style={{ gridTemplateColumns: `auto repeat(${models.length}, 1fr)` }}
            >
                <div />
                {models.map(name => (
                    <div key={`col-${name}`} className={styles.correlationHeader}>
                        {name}
                    </div>
                ))}
                {models.map((rowName, rowIdx) => (
                    <div key={`row-group-${rowName}`} style={{ display: 'contents' }}>
                        <div className={styles.rowHeader}>
                            {rowName}
                        </div>
                        {models.map((colName, colIdx) => {
                            const value = matrix[rowIdx][colIdx];
                            return (
                                <div
                                    key={`${rowName}-${colName}`}
                                    className={styles.correlationCell}
                                    style={{ background: getCorrelationColor(value) }}
                                    title={`${rowName} vs ${colName}: ${value.toFixed(4)}`}
                                >
                                    {value.toFixed(2)}
                                </div>
                            );
                        })}
                    </div>
                ))}
            </div>
        </div>
    );
}
