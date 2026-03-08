import { useState, useEffect } from 'react';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useProject } from '../../hooks/useProject';
import { RunSelector } from './RunSelector';
import { MetricSummary } from './MetricSummary';
import { CalibrationPlot } from './CalibrationPlot';
import { CorrelationMatrix } from './CorrelationMatrix';
import { FoldBreakdown } from './FoldBreakdown';
import { MetaCoefficients } from './MetaCoefficients';
import { ResidualPlot, ResidualStats } from './ResidualPlot';
import { RunComparison } from './RunComparison';
import { PickAnalysis } from './PickAnalysis';
import styles from './Diagnostics.module.css';

interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    has_report?: boolean;
    experiment_id?: string | null;
}

export function Diagnostics() {
    const { events } = useLayoutContext();
    const project = useProject();
    const runsKey = useRefreshKey(events, ['experiments', 'pipeline']);
    const { data: runs, loading, error } = useApi<RunSummary[]>('/api/runs', runsKey, project);
    const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
    const [compareMode, setCompareMode] = useState(false);

    useEffect(() => {
        if (runs && runs.length > 0 && selectedRunIds.length === 0) {
            setSelectedRunIds([runs[0].id]);
        }
    }, [runs, selectedRunIds.length]);

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

    const primaryRunId = selectedRunIds[0] ?? null;
    const selectedRun = runs.find(r => r.id === primaryRunId);
    const metricKeys = selectedRun ? Object.keys(selectedRun.metrics) : [];
    const isRegression = metricKeys.some(k => ['rmse', 'mae', 'r_squared', 'mape'].includes(k));
    const hasMetaCoefficients = selectedRun?.meta_coefficients != null
        && Object.keys(selectedRun.meta_coefficients).length > 0;
    const showComparison = compareMode && selectedRunIds.length >= 2;

    const handleToggleCompare = () => {
        setCompareMode(prev => !prev);
        if (compareMode && selectedRunIds.length > 0) {
            setSelectedRunIds([selectedRunIds[0]]);
        }
    };

    return (
        <div className={styles.diagnostics}>
            <RunSelector
                runs={runs}
                selectedRunIds={selectedRunIds}
                onSelect={setSelectedRunIds}
                compareMode={compareMode}
                onToggleCompare={handleToggleCompare}
            />

            {primaryRunId && (
                <div className={styles.content}>
                    {showComparison && (
                        <section className={styles.section}>
                            <RunComparison runIds={selectedRunIds} runs={runs} />
                        </section>
                    )}

                    {/* Headline metrics */}
                    <section className={styles.section}>
                        <MetricSummary runId={primaryRunId} />
                    </section>

                    {/* Ensemble + secondary stats side by side */}
                    <section className={styles.section}>
                        <div className={styles.twoColumn}>
                            <div>
                                {hasMetaCoefficients && (
                                    <MetaCoefficients runId={primaryRunId} />
                                )}
                                <CorrelationMatrix runId={primaryRunId} />
                            </div>
                            <div>
                                {isRegression ? (
                                    <ResidualStats runId={primaryRunId} />
                                ) : (
                                    <CalibrationPlot runId={primaryRunId} />
                                )}
                            </div>
                        </div>
                    </section>

                    {/* Charts */}
                    {isRegression && (
                        <section className={styles.section}>
                            <ResidualPlot runId={primaryRunId} />
                        </section>
                    )}

                    {/* Pick analysis (only renders if pick_log exists) */}
                    <section className={styles.section}>
                        <PickAnalysis runId={primaryRunId} />
                    </section>

                    {/* Per-fold breakdown */}
                    <section className={styles.section}>
                        <h2 className={styles.sectionTitle}>Per-Fold Breakdown</h2>
                        <FoldBreakdown runId={primaryRunId} />
                    </section>
                </div>
            )}
        </div>
    );
}
