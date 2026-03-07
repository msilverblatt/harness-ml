import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    CartesianGrid, Legend,
} from 'recharts';
import styles from './Diagnostics.module.css';

interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    experiment_id?: string | null;
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

const RUN_COLORS = ['#58a6ff', '#3fb950', '#f0883e'];

function formatTimestampShort(id: string): string {
    const match = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (!match) return id;
    const [, , month, day, hour, minute] = match;
    const months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${months[parseInt(month)]} ${parseInt(day)} ${hour}:${minute}`;
}

function runLabel(id: string, runs: RunSummary[]): string {
    const run = runs.find(r => r.id === id);
    const ts = formatTimestampShort(id);
    return run?.experiment_id ? `${ts} (${run.experiment_id})` : ts;
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

function MetricBarChart({ metricName, orderedRunIds, runData, runs }: {
    metricName: string;
    orderedRunIds: string[];
    runData: Record<string, Record<string, number>>;
    runs: RunSummary[];
}) {
    const chartData = orderedRunIds.map((id, i) => ({
        name: runLabel(id, runs),
        value: runData[id]?.[metricName] ?? 0,
        fill: RUN_COLORS[i % RUN_COLORS.length],
    }));

    const values = chartData.map(d => d.value).filter(v => v !== 0);
    if (values.length === 0) return null;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || max * 0.1 || 0.01;
    const domainMin = Math.max(0, min - range * 0.5);
    const domainMax = max + range * 0.2;

    return (
        <div className={styles.chartContainer} style={{ marginTop: 'var(--space-3)' }}>
            <div className={styles.categoryHeader}>{metricName}</div>
            <ResponsiveContainer width="100%" height={160}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30, top: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" horizontal={false} />
                    <XAxis
                        type="number"
                        domain={[domainMin, domainMax]}
                        tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
                        tickFormatter={(v: number) => formatValue(v)}
                    />
                    <YAxis
                        type="category"
                        dataKey="name"
                        width={140}
                        tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border)',
                            borderRadius: '4px',
                            fontFamily: 'var(--font-mono)',
                            fontSize: '12px',
                        }}
                        formatter={(value: number) => [formatValue(value), metricName]}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {chartData.map((entry, i) => (
                            <rect key={i} fill={entry.fill} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
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
        <div style={{ marginTop: 'var(--space-4)' }}>
            {/* Bar charts per metric */}
            <div className={styles.categoryHeader} style={{ marginTop: 0 }}>
                Metric Comparison
            </div>
            {metricNames.map(metric => (
                <MetricBarChart
                    key={metric}
                    metricName={metric}
                    orderedRunIds={orderedRunIds}
                    runData={runData}
                    runs={runs}
                />
            ))}

            {/* Detail table */}
            <div className={styles.chartContainer} style={{ marginTop: 'var(--space-4)' }}>
                <div className={styles.categoryHeader}>
                    Detail -- first selected run is baseline
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
        </div>
    );
}
