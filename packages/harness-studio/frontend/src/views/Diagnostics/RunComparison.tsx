import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    experiment_id?: string | null;
}

interface MetricsResponse {
    metrics?: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    error?: string;
}

interface RunComparisonProps {
    runIds: string[];
    runs: RunSummary[];
}

const LOWER_IS_BETTER: Record<string, boolean> = {
    brier: true,
    rmse: true,
    mae: true,
    mape: true,
    log_loss: true,
    ece: true,
};

function formatTimestampShort(id: string): string {
    const match = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (!match) return id;
    const [, , month, day, hour, minute] = match;
    const months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${months[parseInt(month)]} ${parseInt(day)} ${hour}:${minute}`;
}

function formatValue(value: number): string {
    if (Math.abs(value) >= 100) return value.toFixed(1);
    if (Math.abs(value) >= 1) return value.toFixed(4);
    return value.toFixed(6);
}

function ComparisonRow({ metricName, values, baselineIdx }: {
    metricName: string;
    values: (number | undefined)[];
    baselineIdx: number;
}) {
    const baseline = values[baselineIdx];
    const lowerBetter = LOWER_IS_BETTER[metricName] ?? false;

    return (
        <tr>
            <td className={styles.compareMetricName}>{metricName}</td>
            {values.map((val, i) => (
                <td key={i} className={styles.compareValue}>
                    {val !== undefined ? formatValue(val) : '--'}
                </td>
            ))}
            {values.length === 2 && baseline !== undefined && values[1] !== undefined && (
                <td className={styles.compareDelta}>
                    <DeltaCell
                        delta={values[1]! - baseline}
                        lowerBetter={lowerBetter}
                    />
                </td>
            )}
            {values.length === 3 && values.slice(1).map((val, i) => {
                if (baseline === undefined || val === undefined) {
                    return <td key={`d${i}`} className={styles.compareDelta}>--</td>;
                }
                return (
                    <td key={`d${i}`} className={styles.compareDelta}>
                        <DeltaCell delta={val - baseline} lowerBetter={lowerBetter} />
                    </td>
                );
            })}
        </tr>
    );
}

function DeltaCell({ delta, lowerBetter }: { delta: number; lowerBetter: boolean }) {
    const improved = lowerBetter ? delta < 0 : delta > 0;
    const color = Math.abs(delta) < 1e-9
        ? 'var(--color-text-muted)'
        : improved
            ? 'var(--color-success)'
            : 'var(--color-error)';
    const sign = delta > 0 ? '+' : '';
    return (
        <span style={{ color, fontFamily: 'var(--font-mono)' }}>
            {sign}{formatValue(delta)}
        </span>
    );
}

export function RunComparison({ runIds, runs }: RunComparisonProps) {
    const allMetricNames = new Set<string>();
    const runData: Record<string, Record<string, number>> = {};

    for (const run of runs) {
        if (runIds.includes(run.id)) {
            runData[run.id] = run.metrics;
            for (const key of Object.keys(run.metrics)) {
                allMetricNames.add(key);
            }
        }
    }

    const metricNames = Array.from(allMetricNames);
    const orderedRunIds = runIds.slice(0, 3);

    return (
        <div className={styles.chartContainer} style={{ marginTop: 'var(--space-4)' }}>
            <div className={styles.categoryHeader}>
                Run Comparison -- first selected run is baseline
            </div>
            <table className={styles.foldTable}>
                <thead>
                    <tr>
                        <th>Metric</th>
                        {orderedRunIds.map(id => {
                            const run = runs.find(r => r.id === id);
                            return (
                                <th key={id}>
                                    {formatTimestampShort(id)}
                                    {run?.experiment_id && (
                                        <span className={styles.expBadgeSmall}>
                                            {run.experiment_id}
                                        </span>
                                    )}
                                </th>
                            );
                        })}
                        {orderedRunIds.length === 2 && <th>Delta</th>}
                        {orderedRunIds.length === 3 && orderedRunIds.slice(1).map((id, i) => (
                            <th key={`delta-${i}`}>vs {formatTimestampShort(id)}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {metricNames.map(metric => (
                        <ComparisonRow
                            key={metric}
                            metricName={metric}
                            values={orderedRunIds.map(id => runData[id]?.[metric])}
                            baselineIdx={0}
                        />
                    ))}
                </tbody>
            </table>
        </div>
    );
}
