import { useApi } from '../../hooks/useApi';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
    ReferenceLine,
    BarChart,
    Bar,
} from 'recharts';
import styles from './Diagnostics.module.css';

interface ResidualResponse {
    scatter?: { predicted: number; actual: number; residual: number }[];
    histogram?: { bin_center: number; count: number }[];
    stats?: {
        mean_residual: number;
        std_residual: number;
        median_residual: number;
        max_overpredict: number;
        max_underpredict: number;
    };
    error?: string;
}

interface ResidualPlotProps {
    runId: string;
}

export function ResidualPlot({ runId }: ResidualPlotProps) {
    const { data, loading, error } = useApi<ResidualResponse>(`/api/runs/${runId}/residuals`);

    if (loading) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>Loading residual analysis...</div>
            </div>
        );
    }

    if (error || !data || data.error || !data.scatter) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>
                    {data?.error ?? error ?? 'No residual data available.'}
                </div>
            </div>
        );
    }

    const { scatter, histogram, stats } = data;

    return (
        <div>
            {stats && (
                <div className={styles.metricsGrid} style={{ marginBottom: 'var(--space-4)' }}>
                    <div className={styles.metricCard}>
                        <div className={styles.metricName}>Mean Residual</div>
                        <div className={styles.metricValue}>{stats.mean_residual.toFixed(1)}</div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricName}>Std Residual</div>
                        <div className={styles.metricValue}>{stats.std_residual.toFixed(1)}</div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricName}>Median Residual</div>
                        <div className={styles.metricValue}>{stats.median_residual.toFixed(1)}</div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricName}>Max Overpredict</div>
                        <div className={styles.metricValue}>{stats.max_overpredict.toFixed(1)}</div>
                    </div>
                    <div className={styles.metricCard}>
                        <div className={styles.metricName}>Max Underpredict</div>
                        <div className={styles.metricValue}>{stats.max_underpredict.toFixed(1)}</div>
                    </div>
                </div>
            )}

            <div className={styles.twoColumn}>
                <div className={styles.chartContainer}>
                    <div className={styles.categoryHeader}>Predicted vs Actual</div>
                    <ResponsiveContainer width="100%" height={300}>
                        <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
                            <XAxis
                                dataKey="predicted"
                                type="number"
                                name="Predicted"
                                tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                                label={{ value: 'Predicted', position: 'insideBottom', offset: -10, fill: 'var(--color-text-secondary)', fontSize: 12 }}
                            />
                            <YAxis
                                dataKey="actual"
                                type="number"
                                name="Actual"
                                tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                                label={{ value: 'Actual', angle: -90, position: 'insideLeft', offset: -5, fill: 'var(--color-text-secondary)', fontSize: 12 }}
                            />
                            <Tooltip
                                contentStyle={{
                                    background: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border)',
                                    borderRadius: 'var(--radius-sm)',
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: 12,
                                    color: 'var(--color-text-primary)',
                                }}
                            />
                            <ReferenceLine
                                segment={[
                                    { x: Math.min(...scatter!.map(s => s.predicted)), y: Math.min(...scatter!.map(s => s.predicted)) },
                                    { x: Math.max(...scatter!.map(s => s.predicted)), y: Math.max(...scatter!.map(s => s.predicted)) },
                                ]}
                                stroke="var(--color-text-muted)"
                                strokeDasharray="6 4"
                            />
                            <Scatter data={scatter} fill="var(--color-accent)" fillOpacity={0.5} r={3} />
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>

                {histogram && histogram.length > 0 && (
                    <div className={styles.chartContainer}>
                        <div className={styles.categoryHeader}>Residual Distribution</div>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={histogram} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
                                <XAxis
                                    dataKey="bin_center"
                                    tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                                    label={{ value: 'Residual', position: 'insideBottom', offset: -10, fill: 'var(--color-text-secondary)', fontSize: 12 }}
                                    tickFormatter={(v: number) => v >= 1000 || v <= -1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}
                                />
                                <YAxis
                                    tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                                    label={{ value: 'Count', angle: -90, position: 'insideLeft', offset: -5, fill: 'var(--color-text-secondary)', fontSize: 12 }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: 'var(--color-bg-tertiary)',
                                        border: '1px solid var(--color-border)',
                                        borderRadius: 'var(--radius-sm)',
                                        fontFamily: 'var(--font-mono)',
                                        fontSize: 12,
                                        color: 'var(--color-text-primary)',
                                    }}
                                    formatter={(value) => [Number(value), 'Count']}
                                    labelFormatter={(label) => `Residual: ${Number(label).toFixed(0)}`}
                                />
                                <ReferenceLine x={0} stroke="var(--color-text-muted)" strokeDasharray="4 4" />
                                <Bar dataKey="count" fill="var(--color-accent)" fillOpacity={0.7} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>
        </div>
    );
}
