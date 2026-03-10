import { useState, useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useProject } from '../../hooks/useProject';
import { EmptyState } from '../../components/EmptyState/EmptyState';
import { ExperimentTable } from './ExperimentTable';
import { MetricChart } from './MetricChart';
import { ComparePanel } from './ComparePanel';
import type { Experiment } from './ExperimentTable';
import styles from './Experiments.module.css';

export function Experiments() {
    const { events } = useLayoutContext();
    const project = useProject();
    const location = useLocation();
    const initialExpandedId = (location.state as { expandedId?: string } | null)?.expandedId ?? null;
    const expKey = useRefreshKey(events, ['experiments', 'pipeline']);
    const { data: experiments, loading, error } = useApi<Experiment[]>('/api/experiments', expKey, project);
    const [compareIds, setCompareIds] = useState<string[]>([]);
    const [chartMetric, setChartMetric] = useState<string | null>(null);

    const availableMetrics = useMemo(() => {
        if (!experiments) return [];
        const keys = new Set<string>();
        for (const exp of experiments) {
            if (exp.metrics) Object.keys(exp.metrics).forEach(k => keys.add(k));
        }
        return Array.from(keys).sort();
    }, [experiments]);

    const activeMetric = useMemo(() => {
        if (chartMetric && availableMetrics.includes(chartMetric)) return chartMetric;
        if (!experiments) return null;
        for (const exp of experiments) {
            if (exp.primary_metric) return exp.primary_metric;
        }
        return availableMetrics[0] ?? null;
    }, [experiments, chartMetric, availableMetrics]);

    function handleToggleCompare(id: string) {
        setCompareIds(prev => {
            if (prev.includes(id)) {
                return prev.filter(x => x !== id);
            }
            return [...prev, id];
        });
    }

    if (loading) {
        return (
            <div className={styles.experiments}>
                <div className={styles.emptyState}>Loading experiments...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.experiments}>
                <div className={styles.emptyState}>Error: {error}</div>
            </div>
        );
    }

    const exps = experiments ?? [];

    if (exps.length === 0) {
        return (
            <div className={styles.experiments}>
                <EmptyState
                    title="No experiments recorded"
                    description="Create an experiment to start tracking hypotheses, metrics, and conclusions."
                />
            </div>
        );
    }

    return (
        <div className={styles.experiments}>
            {activeMetric && (
                <div className={styles.chartSection}>
                    <MetricChart
                        experiments={exps}
                        metricKey={activeMetric}
                        availableMetrics={availableMetrics}
                        onMetricChange={setChartMetric}
                    />
                </div>
            )}
            <div className={styles.tableSection}>
                <ExperimentTable
                    experiments={exps}
                    selectedIds={compareIds}
                    onToggleCompare={handleToggleCompare}
                    initialExpandedId={initialExpandedId}
                    allMetricKeys={availableMetrics}
                />
            </div>
            {compareIds.length >= 2 && (
                <ComparePanel experiments={exps} selectedIds={compareIds} />
            )}
        </div>
    );
}
