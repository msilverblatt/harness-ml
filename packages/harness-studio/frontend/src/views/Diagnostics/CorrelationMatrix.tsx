import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import { useTheme } from '../../hooks/useTheme';
import type { ThemeColors } from '../../styles/colors';
import { Tooltip } from '../../components/Tooltip/Tooltip';
import styles from './Diagnostics.module.css';

interface CorrelationResponse {
    models?: string[];
    matrix?: number[][];
    error?: string;
}

interface CorrelationMatrixProps {
    runId: string;
}

function getCorrelationColor(value: number, colors: ThemeColors): string {
    if (value >= 0.99) return 'var(--color-text-muted)';
    if (value > 0.8) return colors.error + '66';
    if (value > 0.6) return colors.warning + '66';
    return colors.success + '66';
}

export function CorrelationMatrix({ runId }: CorrelationMatrixProps) {
    const { colors } = useTheme();
    const project = useProject();
    const { data, loading, error } = useApi<CorrelationResponse>(`/api/runs/${runId}/correlations`, undefined, project);

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
            <div className={styles.categoryHeader}><Tooltip term="correlation" label="Prediction Correlations" /></div>
            <table className={styles.corrTable}>
                <thead>
                    <tr>
                        <th />
                        {models.map(name => (
                            <th key={`col-${name}`} className={styles.corrColHeader}>
                                {name}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {models.map((rowName, rowIdx) => (
                        <tr key={rowName}>
                            <td className={styles.corrRowHeader}>{rowName}</td>
                            {models.map((colName, colIdx) => {
                                const value = matrix[rowIdx][colIdx];
                                return (
                                    <td
                                        key={`${rowName}-${colName}`}
                                        className={styles.corrCell}
                                        style={{ background: getCorrelationColor(value, colors) }}
                                        title={`${rowName} vs ${colName}: ${value.toFixed(4)}`}
                                    >
                                        {value.toFixed(2)}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
