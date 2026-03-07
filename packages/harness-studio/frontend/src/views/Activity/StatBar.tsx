import { StatBox } from '../../components/StatBox/StatBox';
import { useApi } from '../../hooks/useApi';
import styles from './Activity.module.css';

interface ProjectStatus {
    project_name: string;
    task: string;
    model_types_tried: number;
    active_models: number;
    experiments_run: number;
    run_count: number;
    feature_count: number;
    latest_metrics: Record<string, number>;
}

interface StatBarProps {
    connected: boolean;
}

export function StatBar({ connected }: StatBarProps) {
    const { data: status } = useApi<ProjectStatus>('/api/project/status');

    const MAX_METRICS = 3;
    const metricEntries: { label: string; value: string }[] = [];
    if (status?.latest_metrics) {
        const keys = Object.keys(status.latest_metrics);
        for (let idx = 0; idx < Math.min(keys.length, MAX_METRICS); idx++) {
            const key = keys[idx];
            const val = status.latest_metrics[key];
            metricEntries.push({
                label: key,
                value: typeof val === 'number'
                    ? (Math.abs(val) >= 100 ? val.toFixed(1) : val.toFixed(4))
                    : String(val),
            });
        }
    }

    return (
        <div className={styles.statBar}>
            <StatBox
                label="Project"
                value={status?.project_name ?? '\u2014'}
                subtitle={status?.task}
            />
            <StatBox
                label="Features"
                value={status?.feature_count ?? 0}
            />
            <StatBox
                label="Models"
                value={status?.active_models ?? 0}
                subtitle={`${status?.model_types_tried ?? 0} types`}
            />
            <StatBox
                label="Experiments"
                value={status?.experiments_run ?? 0}
            />
            <StatBox
                label="Runs"
                value={status?.run_count ?? 0}
            />
            {metricEntries.map(entry => (
                <StatBox
                    key={entry.label}
                    label={entry.label}
                    value={entry.value}
                    variant="success"
                />
            ))}
            <StatBox
                label="WebSocket"
                value={connected ? 'Live' : 'Off'}
                variant={connected ? 'success' : 'default'}
            />
        </div>
    );
}
