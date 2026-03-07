import type { Experiment } from './ExperimentTable';
import styles from './Experiments.module.css';

interface ComparePanelProps {
    experiments: Experiment[];
    selectedIds: string[];
}

// Metrics where lower is better
const LOWER_IS_BETTER = new Set(['brier', 'ece', 'log_loss', 'mae', 'mse', 'rmse']);

function deltaClass(metricName: string, delta: number): string {
    if (delta === 0) return '';
    const lowerBetter = LOWER_IS_BETTER.has(metricName);
    const isImprovement = lowerBetter ? delta < 0 : delta > 0;
    return isImprovement ? styles.deltaPositive : styles.deltaNegative;
}

function formatValue(v: number | undefined): string {
    if (v == null) return '--';
    return v.toFixed(4);
}

export function ComparePanel({ experiments, selectedIds }: ComparePanelProps) {
    if (selectedIds.length !== 2) return null;

    const expA = experiments.find(e => e.experiment_id === selectedIds[0]);
    const expB = experiments.find(e => e.experiment_id === selectedIds[1]);

    if (!expA || !expB) return null;

    // Collect all metric keys from both experiments
    const allKeys = new Set<string>();
    if (expA.metrics) Object.keys(expA.metrics).forEach(k => allKeys.add(k));
    if (expB.metrics) Object.keys(expB.metrics).forEach(k => allKeys.add(k));

    const metricKeys = Array.from(allKeys).sort();

    if (metricKeys.length === 0) {
        return (
            <div className={styles.comparePanel}>
                <div className={styles.compareTitle}>No metrics to compare</div>
            </div>
        );
    }

    return (
        <div className={styles.comparePanel}>
            <div className={styles.compareTitle}>
                Comparing: {expA.experiment_id} vs {expB.experiment_id}
            </div>
            <table className={styles.compareTable}>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>{expA.experiment_id}</th>
                        <th>{expB.experiment_id}</th>
                        <th>Delta</th>
                    </tr>
                </thead>
                <tbody>
                    {metricKeys.map(key => {
                        const valA = expA.metrics?.[key];
                        const valB = expB.metrics?.[key];
                        const delta = valA != null && valB != null ? valB - valA : undefined;
                        const sign = delta != null && delta > 0 ? '+' : '';
                        return (
                            <tr key={key}>
                                <td>{key}</td>
                                <td>{formatValue(valA)}</td>
                                <td>{formatValue(valB)}</td>
                                <td className={delta != null ? deltaClass(key, delta) : ''}>
                                    {delta != null ? `${sign}${delta.toFixed(4)}` : '--'}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
