import { useState, useEffect } from 'react';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import { RunSelector } from './RunSelector';
import { MetricSummary } from './MetricSummary';
import { CalibrationPlot } from './CalibrationPlot';
import { CorrelationMatrix } from './CorrelationMatrix';
import { FoldBreakdown } from './FoldBreakdown';
import { MetaCoefficients } from './MetaCoefficients';
import { ResidualPlot } from './ResidualPlot';
import { RunComparison } from './RunComparison';
import { ReportView } from './ReportView';
import styles from './Diagnostics.module.css';

interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    has_report?: boolean;
    experiment_id?: string | null;
}

type TabId = 'overview' | 'report' | 'compare';

export function Diagnostics() {
    const { events } = useLayoutContext();
    const runsKey = useRefreshKey(events, ['experiments', 'pipeline']);
    const { data: runs, loading, error } = useApi<RunSummary[]>('/api/runs', runsKey);
    const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
    const [compareMode, setCompareMode] = useState(false);
    const [activeTab, setActiveTab] = useState<TabId>('overview');

    useEffect(() => {
        if (runs && runs.length > 0 && selectedRunIds.length === 0) {
            setSelectedRunIds([runs[0].id]);
        }
    }, [runs, selectedRunIds.length]);

    useEffect(() => {
        if (compareMode && selectedRunIds.length >= 2) {
            setActiveTab('compare');
        } else if (activeTab === 'compare' && selectedRunIds.length < 2) {
            setActiveTab('overview');
        }
    }, [selectedRunIds.length, compareMode]);

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
    const hasReport = selectedRun?.has_report ?? false;
    const showCompareTab = selectedRunIds.length >= 2;

    const handleToggleCompare = () => {
        setCompareMode(prev => !prev);
        if (compareMode) {
            if (selectedRunIds.length > 0) {
                setSelectedRunIds([selectedRunIds[0]]);
            }
            setActiveTab('overview');
        }
    };

    const handleSelect = (ids: string[]) => {
        setSelectedRunIds(ids);
        if (!compareMode && ids.length === 1) {
            setActiveTab('overview');
        }
    };

    const tabs: { id: TabId; label: string; visible: boolean }[] = [
        { id: 'overview', label: 'Overview', visible: true },
        { id: 'report', label: 'Report', visible: hasReport },
        { id: 'compare', label: 'Compare', visible: showCompareTab },
    ];

    return (
        <div className={styles.diagnostics}>
            <RunSelector
                runs={runs}
                selectedRunIds={selectedRunIds}
                onSelect={handleSelect}
                compareMode={compareMode}
                onToggleCompare={handleToggleCompare}
            />

            {primaryRunId && (
                <>
                    <div className={styles.tabBar}>
                        {tabs.filter(t => t.visible).map(tab => (
                            <button
                                key={tab.id}
                                className={`${styles.tab} ${activeTab === tab.id ? styles.tabActive : ''}`}
                                onClick={() => setActiveTab(tab.id)}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    {activeTab === 'overview' && (
                        <>
                            <div className={styles.sectionTitle}>Metrics</div>
                            <MetricSummary runId={primaryRunId} />

                            <div className={styles.sectionTitle}>Per-Fold Breakdown</div>
                            <FoldBreakdown runId={primaryRunId} />

                            {hasMetaCoefficients ? (
                                <div className={styles.twoColumn}>
                                    <div>
                                        <div className={styles.sectionTitle}>Model Weights</div>
                                        <MetaCoefficients runId={primaryRunId} />
                                    </div>
                                    <div>
                                        <div className={styles.sectionTitle}>Model Correlations</div>
                                        <CorrelationMatrix runId={primaryRunId} />
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <div className={styles.sectionTitle}>Model Correlations</div>
                                    <CorrelationMatrix runId={primaryRunId} />
                                </>
                            )}

                            <div className={styles.sectionTitle}>
                                {isRegression ? 'Predicted vs Actual' : 'Calibration'}
                            </div>
                            {isRegression ? (
                                <ResidualPlot runId={primaryRunId} />
                            ) : (
                                <CalibrationPlot runId={primaryRunId} />
                            )}
                        </>
                    )}

                    {activeTab === 'report' && hasReport && (
                        <ReportView runId={primaryRunId} />
                    )}

                    {activeTab === 'compare' && selectedRunIds.length >= 2 && (
                        <RunComparison runIds={selectedRunIds} runs={runs} />
                    )}
                </>
            )}
        </div>
    );
}
