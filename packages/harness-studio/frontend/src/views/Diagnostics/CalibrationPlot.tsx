import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import {
    ComposedChart,
    Scatter,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
    ReferenceLine,
} from 'recharts';
import { Tooltip as GlossaryTip } from '../../components/Tooltip/Tooltip';
import styles from './Diagnostics.module.css';

interface CalibrationPoint {
    bin_center: number;
    predicted: number;
    actual: number;
    count: number;
}

interface CalibrationResponse {
    calibration?: CalibrationPoint[];
    prob_column?: string;
    error?: string;
}

interface CalibrationPlotProps {
    runId: string;
}

export function CalibrationPlot({ runId }: CalibrationPlotProps) {
    const project = useProject();
    const { data, loading, error } = useApi<CalibrationResponse>(`/api/runs/${runId}/calibration`, undefined, project);

    if (loading) {
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>Loading calibration data...</div>
            </div>
        );
    }

    if (error || !data || data.error || !data.calibration || data.calibration.length === 0) {
        const msg = data?.error ?? error ?? 'No calibration data available.';
        return (
            <div className={styles.chartContainer}>
                <div className={styles.emptyState}>{msg}</div>
            </div>
        );
    }

    const plotData = data.calibration.map(point => ({
        predicted: point.predicted,
        actual: point.actual,
        count: point.count,
    }));

    return (
        <div className={styles.chartContainer}>
            <div className={styles.categoryHeader}>
                <GlossaryTip term="calibration" label="Calibration Curve" /> ({data.prob_column ?? 'ensemble'})
            </div>
            <ResponsiveContainer width="100%" height={300}>
                <ComposedChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="var(--color-border-subtle)"
                    />
                    <XAxis
                        dataKey="predicted"
                        type="number"
                        domain={[0, 1]}
                        tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                        label={{ value: 'Predicted', position: 'insideBottom', offset: -10, fill: 'var(--color-text-secondary)', fontSize: 12 }}
                    />
                    <YAxis
                        dataKey="actual"
                        type="number"
                        domain={[0, 1]}
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
                        formatter={(value) => [Number(value).toFixed(4), 'Actual']}
                        labelFormatter={(label) => `Predicted: ${Number(label).toFixed(4)}`}
                    />
                    <ReferenceLine
                        segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                        stroke="var(--color-text-muted)"
                        strokeDasharray="6 4"
                        strokeWidth={1}
                    />
                    <Scatter
                        name="Actual"
                        data={plotData}
                        fill="var(--color-accent)"
                        r={5}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}
