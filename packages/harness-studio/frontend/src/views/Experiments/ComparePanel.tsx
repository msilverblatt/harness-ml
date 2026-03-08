import type { Experiment } from './ExperimentTable';
import { MetricLabel } from '../../components/Tooltip/Tooltip';
import styles from './Experiments.module.css';

interface ComparePanelProps {
    experiments: Experiment[];
    selectedIds: string[];
}

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
    if (selectedIds.length < 2) return null;

    const selected = selectedIds
        .map(id => experiments.find(e => e.experiment_id === id))
        .filter((e): e is Experiment => e != null);

    if (selected.length < 2) return null;

    const allKeys = new Set<string>();
    for (const exp of selected) {
        if (exp.metrics) Object.keys(exp.metrics).forEach(k => allKeys.add(k));
    }
    const metricKeys = Array.from(allKeys).sort();

    if (metricKeys.length === 0) {
        return (
            <div className={styles.comparePanel}>
                <div className={styles.compareTitle}>No metrics to compare</div>
            </div>
        );
    }

    const firstExp = selected[0];

    return (
        <div className={styles.comparePanel}>
            <div className={styles.compareTitle}>
                Comparing: {selected.map(e => e.experiment_id).join(' / ')}
            </div>
            <div className={styles.compareTableWrap}>
                <table className={styles.compareTable}>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            {selected.map(exp => (
                                <th key={exp.experiment_id}>{exp.experiment_id}</th>
                            ))}
                            {selected.length >= 2 && selected.slice(1).map(exp => (
                                <th key={`delta-${exp.experiment_id}`}>
                                    vs {firstExp.experiment_id}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {metricKeys.map(key => {
                            const baseVal = firstExp.metrics?.[key];
                            return (
                                <tr key={key}>
                                    <td><MetricLabel name={key} /></td>
                                    {selected.map(exp => (
                                        <td key={exp.experiment_id}>
                                            {formatValue(exp.metrics?.[key])}
                                        </td>
                                    ))}
                                    {selected.slice(1).map(exp => {
                                        const val = exp.metrics?.[key];
                                        const delta = val != null && baseVal != null ? val - baseVal : undefined;
                                        const sign = delta != null && delta > 0 ? '+' : '';
                                        return (
                                            <td
                                                key={`delta-${exp.experiment_id}`}
                                                className={delta != null ? deltaClass(key, delta) : ''}
                                            >
                                                {delta != null ? `${sign}${delta.toFixed(4)}` : '--'}
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
