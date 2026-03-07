import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, CartesianGrid } from 'recharts';
import type { Experiment } from './ExperimentTable';
import styles from './Experiments.module.css';

interface MetricChartProps {
    experiments: Experiment[];
    metricKey: string;
    availableMetrics: string[];
    onMetricChange: (metric: string) => void;
}

const LOWER_IS_BETTER = new Set(['brier', 'ece', 'log_loss', 'mae', 'mse', 'rmse']);

export function MetricChart({ experiments, metricKey, availableMetrics, onMetricChange }: MetricChartProps) {
    const chronological = [...experiments].reverse();

    const isLowerBetter = LOWER_IS_BETTER.has(metricKey);

    const data = useMemo(() => {
        const points = chronological
            .filter(exp => exp.metrics && exp.metrics[metricKey] != null)
            .map(exp => ({
                id: exp.experiment_id,
                value: exp.metrics![metricKey],
            }));

        let runningBest: number | null = null;
        return points.map(p => {
            if (runningBest === null) {
                runningBest = p.value;
            } else {
                runningBest = isLowerBetter
                    ? Math.min(runningBest, p.value)
                    : Math.max(runningBest, p.value);
            }
            const isNewBest = p.value === runningBest;
            return { ...p, isNewBest };
        });
    }, [chronological, metricKey, isLowerBetter]);

    if (data.length === 0) {
        return null;
    }

    const withBaseline = chronological.find(
        exp => exp.baseline_metrics && exp.baseline_metrics[metricKey] != null
    );
    const baselineValue = withBaseline?.baseline_metrics?.[metricKey];

    const allValues = data.map(d => d.value);
    const bestValue = isLowerBetter ? Math.min(...allValues) : Math.max(...allValues);

    if (baselineValue != null) allValues.push(baselineValue);
    allValues.push(bestValue);
    const yMin = Math.min(...allValues);
    const yMax = Math.max(...allValues);
    const padding = (yMax - yMin) * 0.15 || 0.01;

    return (
        <div className={styles.chartContainer}>
            <div className={styles.metricSelector}>
                {availableMetrics.map(m => (
                    <button
                        key={m}
                        className={`${styles.metricTab} ${m === metricKey ? styles.metricTabActive : ''}`}
                        onClick={() => onMetricChange(m)}
                    >
                        {m}
                    </button>
                ))}
            </div>
            <div className={styles.chartInner}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
                        <CartesianGrid
                            stroke="var(--color-border-subtle)"
                            strokeDasharray="3 3"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="id"
                            tick={{ fill: 'var(--color-text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                            axisLine={{ stroke: 'var(--color-border-subtle)' }}
                            tickLine={false}
                        />
                        <YAxis
                            domain={[yMin - padding, yMax + padding]}
                            tick={{ fill: 'var(--color-text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                            axisLine={{ stroke: 'var(--color-border-subtle)' }}
                            tickLine={false}
                            tickFormatter={(v: number) => v.toFixed(3)}
                        />
                        <Tooltip
                            contentStyle={{
                                background: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border)',
                                borderRadius: 'var(--radius-sm)',
                                color: 'var(--color-text-primary)',
                                fontFamily: 'var(--font-mono)',
                                fontSize: 12,
                            }}
                            labelStyle={{ color: 'var(--color-text-secondary)' }}
                            formatter={(value) => [Number(value).toFixed(4), metricKey]}
                        />
                        {baselineValue != null && (
                            <ReferenceLine
                                y={baselineValue}
                                stroke="var(--color-text-muted)"
                                strokeDasharray="6 3"
                                label={{
                                    value: 'baseline',
                                    fill: 'var(--color-text-muted)',
                                    fontSize: 10,
                                    fontFamily: 'var(--font-mono)',
                                    position: 'right',
                                }}
                            />
                        )}
                        <ReferenceLine
                            y={bestValue}
                            stroke="var(--color-success)"
                            strokeWidth={1.5}
                            label={{
                                value: `best: ${bestValue.toFixed(4)}`,
                                fill: 'var(--color-success)',
                                fontSize: 10,
                                fontFamily: 'var(--font-mono)',
                                position: 'left',
                            }}
                        />
                        <Line
                            type="monotone"
                            dataKey="value"
                            stroke="var(--color-accent)"
                            strokeWidth={2}
                            dot={(props: Record<string, unknown>) => {
                                const { cx, cy, payload } = props as { cx: number; cy: number; payload: { isNewBest: boolean } };
                                if (payload.isNewBest) {
                                    return (
                                        <circle
                                            key={`dot-${cx}-${cy}`}
                                            cx={cx}
                                            cy={cy}
                                            r={5}
                                            fill="#d29922"
                                            stroke="#d29922"
                                            strokeWidth={2}
                                        />
                                    );
                                }
                                return (
                                    <circle
                                        key={`dot-${cx}-${cy}`}
                                        cx={cx}
                                        cy={cy}
                                        r={3}
                                        fill="var(--color-accent)"
                                        stroke="var(--color-accent)"
                                        strokeWidth={1}
                                    />
                                );
                            }}
                            activeDot={{ r: 5 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
