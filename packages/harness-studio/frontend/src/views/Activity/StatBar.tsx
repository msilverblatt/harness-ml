import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
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

function formatMetric(val: number): string {
    if (Math.abs(val) >= 100) return val.toFixed(1);
    if (Math.abs(val) >= 1) return val.toFixed(4);
    return val.toFixed(4);
}

export function StatBar() {
    const project = useProject();
    const { data: status } = useApi<ProjectStatus>('/api/project/status', undefined, project);

    const metricEntries = Object.entries(status?.latest_metrics ?? {});

    return (
        <div className={styles.statBar}>
            <div className={styles.statGrid}>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{status?.feature_count ?? 0}</span>
                    <span className={styles.statLabel}>Features</span>
                </div>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{status?.active_models ?? 0}</span>
                    <span className={styles.statLabel}>Models</span>
                </div>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{status?.experiments_run ?? 0}</span>
                    <span className={styles.statLabel}>Experiments</span>
                </div>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{status?.run_count ?? 0}</span>
                    <span className={styles.statLabel}>Runs</span>
                </div>
                {metricEntries.map(([name, value]) => (
                    <div key={name} className={styles.stat}>
                        <span className={styles.statValue}>{formatMetric(value)}</span>
                        <span className={styles.statLabel}>{name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
