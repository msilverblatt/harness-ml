import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import styles from './Diagnostics.module.css';

interface PickStats {
    total?: number;
    correct?: number;
    accuracy?: number;
    avg_confidence?: number;
    avg_agreement?: number | null;
    avg_confidence_correct?: number;
    avg_confidence_incorrect?: number;
    error?: string;
}

interface PickAnalysisProps {
    runId: string;
}

export function PickAnalysis({ runId }: PickAnalysisProps) {
    const project = useProject();
    const { data, loading, error } = useApi<PickStats>(`/api/runs/${runId}/picks`, undefined, project);

    if (loading) {
        return <div className={styles.emptyState}>Loading pick analysis...</div>;
    }

    if (error || !data || data.error || data.total == null) {
        return null; // silently hide if no pick data
    }

    const stats: { label: string; value: string }[] = [
        { label: 'Total Picks', value: String(data.total) },
        { label: 'Correct', value: String(data.correct) },
        { label: 'Accuracy', value: `${((data.accuracy ?? 0) * 100).toFixed(1)}%` },
        { label: 'Avg Confidence', value: (data.avg_confidence ?? 0).toFixed(4) },
    ];

    if (data.avg_agreement != null) {
        stats.push({ label: 'Model Agreement', value: `${(data.avg_agreement * 100).toFixed(1)}%` });
    }
    if (data.avg_confidence_correct != null) {
        stats.push({ label: 'Conf (Correct)', value: data.avg_confidence_correct.toFixed(4) });
    }
    if (data.avg_confidence_incorrect != null) {
        stats.push({ label: 'Conf (Incorrect)', value: data.avg_confidence_incorrect.toFixed(4) });
    }

    return (
        <div
            className={styles.metricsGrid}
            style={{ '--metric-count': stats.length } as React.CSSProperties}
        >
            {stats.map(s => (
                <div key={s.label} className={styles.metricCard}>
                    <span className={styles.metricValue}>{s.value}</span>
                    <span className={styles.metricName}>{s.label}</span>
                </div>
            ))}
        </div>
    );
}
