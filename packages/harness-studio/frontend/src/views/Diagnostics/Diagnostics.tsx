import { useState, useEffect } from 'react';
import { useApi } from '../../hooks/useApi';
import { RunSelector } from './RunSelector';
import { MetricSummary } from './MetricSummary';
import { CalibrationPlot } from './CalibrationPlot';
import { CorrelationMatrix } from './CorrelationMatrix';
import { FoldBreakdown } from './FoldBreakdown';
import styles from './Diagnostics.module.css';

interface Run {
    id: string;
    metrics: Record<string, number>;
}

export function Diagnostics() {
    const { data: runs, loading, error } = useApi<Run[]>('/api/runs');
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

    useEffect(() => {
        if (runs && runs.length > 0 && selectedRunId === null) {
            setSelectedRunId(runs[0].id);
        }
    }, [runs, selectedRunId]);

    if (loading) {
        return (
            <div className={styles.diagnostics}>
                <div className={styles.emptyState}>Loading...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.diagnostics}>
                <div className={styles.emptyState}>Error: {error}</div>
            </div>
        );
    }

    if (!runs || runs.length === 0) {
        return (
            <div className={styles.diagnostics}>
                <div className={styles.emptyState}>No runs available. Run a pipeline to see diagnostics.</div>
            </div>
        );
    }

    return (
        <div className={styles.diagnostics}>
            <RunSelector
                selectedRunId={selectedRunId}
                onSelect={setSelectedRunId}
            />

            {selectedRunId && (
                <>
                    <div className={styles.sectionTitle}>Metrics</div>
                    <MetricSummary runId={selectedRunId} />

                    <div className={styles.twoColumn}>
                        <CalibrationPlot runId={selectedRunId} />
                        <CorrelationMatrix runId={selectedRunId} />
                    </div>

                    <div className={styles.sectionTitle}>Fold Analysis</div>
                    <FoldBreakdown runId={selectedRunId} />
                </>
            )}
        </div>
    );
}
