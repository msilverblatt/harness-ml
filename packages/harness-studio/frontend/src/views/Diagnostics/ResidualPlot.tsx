import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
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
import { Tooltip as GlossaryTip } from '../../components/Tooltip/Tooltip';
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

interface ResidualProps {
    runId: string;
}

function useResidualData(runId: string) {
    const project = useProject();
    return useApi<ResidualResponse>(`/api/runs/${runId}/residuals`, undefined, project);
}

export function ResidualStats({ runId }: ResidualProps) {
    const { data, loading, error } = useResidualData(runId);

    if (loading || error || !data || data.error || !data.stats) {
        return null;
    }

    const { stats } = data;
    const items: Array<{ key: string; label: React.ReactNode; value: string }> = [
        { key: 'mean', label: <GlossaryTip term="mean_residual" label="Mean Residual" />, value: stats.mean_residual.toFixed(1) },
        { key: 'std', label: <GlossaryTip term="std_residual" label="Std Residual" />, value: stats.std_residual.toFixed(1) },
        { key: 'median', label: <GlossaryTip term="median_residual" label="Median Residual" />, value: stats.median_residual.toFixed(1) },
        { key: 'over', label: 'Max Overpredict', value: stats.max_overpredict.toFixed(1) },
        { key: 'under', label: 'Max Underpredict', value: stats.max_underpredict.toFixed(1) },
    ];

    return (
        <div className={styles.chartContainer}>
            <div className={styles.categoryHeader}><GlossaryTip term="residual" label="Residual Summary" /></div>
            <div className={styles.statsList}>
                {items.map(s => (
                    <div key={s.key} className={styles.statsRow}>
                        <span className={styles.statsLabel}>{s.label}</span>
                        <span className={styles.statsValue}>{s.value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export function ResidualPlot({ runId }: ResidualProps) {
    const { data, loading, error } = useResidualData(runId);

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

    const { scatter, histogram } = data;

    return (
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
    );
}
