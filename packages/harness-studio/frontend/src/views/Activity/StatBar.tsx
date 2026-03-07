import { StatBox } from '../../components/StatBox/StatBox';
import { useApi } from '../../hooks/useApi';
import styles from './Activity.module.css';

interface EventStats {
    total_calls: number;
    errors: number;
    by_tool: Record<string, number>;
}

interface ProjectStatus {
    phase: string;
    model_types_tried: number;
    experiments_run: number;
    has_baseline: boolean;
    feature_discovery_done: boolean;
}

interface ProjectConfig {
    pipeline: Record<string, unknown>;
    models: Record<string, unknown>;
}

interface StatBarProps {
    connected: boolean;
}

export function StatBar({ connected }: StatBarProps) {
    const { data: stats } = useApi<EventStats>('/api/events/stats');
    const { data: status } = useApi<ProjectStatus>('/api/project/status');
    const { data: config } = useApi<ProjectConfig>('/api/project/config');

    const projectName = (config?.pipeline as Record<string, unknown>)?.name as string | undefined;

    return (
        <div className={styles.statBar}>
            <StatBox
                label="Project"
                value={projectName ?? '\u2014'}
                subtitle={status?.phase}
            />
            <StatBox
                label="Experiments"
                value={status?.experiments_run ?? 0}
            />
            <StatBox
                label="Models Tried"
                value={status?.model_types_tried ?? 0}
            />
            <StatBox
                label="Tool Calls"
                value={stats?.total_calls ?? 0}
            />
            <StatBox
                label="Errors"
                value={stats?.errors ?? 0}
                variant={(stats?.errors ?? 0) > 0 ? 'error' : 'default'}
            />
            <StatBox
                label="WebSocket"
                value={connected ? 'Connected' : 'Disconnected'}
                variant={connected ? 'success' : 'error'}
            />
        </div>
    );
}
