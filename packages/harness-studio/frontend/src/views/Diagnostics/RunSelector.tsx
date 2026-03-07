import { useApi } from '../../hooks/useApi';
import styles from './Diagnostics.module.css';

interface Run {
    id: string;
    metrics: Record<string, number>;
}

interface RunSelectorProps {
    selectedRunId: string | null;
    onSelect: (runId: string) => void;
}

export function RunSelector({ selectedRunId, onSelect }: RunSelectorProps) {
    const { data: runs, loading, error } = useApi<Run[]>('/api/runs');

    if (loading) {
        return (
            <div className={styles.selectorRow}>
                <span className={styles.selectorLabel}>Run:</span>
                <select className={styles.select} disabled>
                    <option>Loading...</option>
                </select>
            </div>
        );
    }

    if (error || !runs || runs.length === 0) {
        return (
            <div className={styles.selectorRow}>
                <span className={styles.selectorLabel}>Run:</span>
                <select className={styles.select} disabled>
                    <option>{error ? `Error: ${error}` : 'No runs available'}</option>
                </select>
            </div>
        );
    }

    return (
        <div className={styles.selectorRow}>
            <span className={styles.selectorLabel}>Run:</span>
            <select
                className={styles.select}
                value={selectedRunId ?? runs[0].id}
                onChange={e => onSelect(e.target.value)}
            >
                {runs.map(run => (
                    <option key={run.id} value={run.id}>
                        {run.id}
                    </option>
                ))}
            </select>
        </div>
    );
}
