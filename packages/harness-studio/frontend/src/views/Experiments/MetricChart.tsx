import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, CartesianGrid } from 'recharts';
import type { Experiment } from './ExperimentTable';

interface MetricChartProps {
    experiments: Experiment[];
    metricKey?: string;
}

export function MetricChart({ experiments, metricKey = 'brier' }: MetricChartProps) {
    // Chronological order (API returns newest first, so reverse)
    const chronological = [...experiments].reverse();

    const data = chronological
        .filter(exp => exp.metrics && exp.metrics[metricKey] != null)
        .map(exp => ({
            id: exp.experiment_id,
            value: exp.metrics![metricKey],
        }));

    if (data.length === 0) {
        return null;
    }

    // Find baseline value from the most recent experiment that has baseline_metrics
    const withBaseline = chronological.find(
        exp => exp.baseline_metrics && exp.baseline_metrics[metricKey] != null
    );
    const baselineValue = withBaseline?.baseline_metrics?.[metricKey];

    // Compute Y domain with some padding
    const values = data.map(d => d.value);
    if (baselineValue != null) values.push(baselineValue);
    const yMin = Math.min(...values);
    const yMax = Math.max(...values);
    const padding = (yMax - yMin) * 0.15 || 0.01;

    return (
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
                        }}
                    />
                )}
                <Line
                    type="monotone"
                    dataKey="value"
                    stroke="var(--color-accent)"
                    strokeWidth={2}
                    dot={{ fill: 'var(--color-accent)', r: 3 }}
                    activeDot={{ r: 5 }}
                />
            </LineChart>
        </ResponsiveContainer>
    );
}
