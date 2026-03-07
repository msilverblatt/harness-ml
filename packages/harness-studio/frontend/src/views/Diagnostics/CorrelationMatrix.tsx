import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface CorrelationResponse {
    models: string[];
    matrix: number[][];
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

    if (error || !data || data.models.length === 0) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>
                    {error ? `Error: ${error}` : 'No correlation data available.'}
                </div>
            </div>
        );
    }

    const { models, matrix } = data;

    return (
        <div className={styles.chartContainer}>
            <div className={styles.categoryHeader}>Model Correlations</div>
            <div
                className={styles.correlationGrid}
                style={{ gridTemplateColumns: `auto repeat(${models.length}, 1fr)` }}
            >
                {/* Top-left empty cell */}
                <div />
                {/* Column headers */}
                {models.map(name => (
                    <div key={`col-${name}`} className={styles.correlationHeader}>
                        {name}
                    </div>
                ))}
                {/* Rows */}
                {models.map((rowName, rowIdx) => (
                    <>
                        <div key={`row-${rowName}`} className={styles.rowHeader}>
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
                    </>
                ))}
            </div>
        </div>
    );
}
