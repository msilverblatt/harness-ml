import { useState } from 'react';
import { useApi } from '../../hooks/useApi';
import { ExperimentTable } from './ExperimentTable';
import { MetricChart } from './MetricChart';
import { ComparePanel } from './ComparePanel';
import type { Experiment } from './ExperimentTable';
import styles from './Experiments.module.css';

export function Experiments() {
    const { data: experiments, loading, error } = useApi<Experiment[]>('/api/experiments');
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [compareIds, setCompareIds] = useState<string[]>([]);

    function handleSelect(id: string) {
        setSelectedId(prev => prev === id ? null : id);
    }

    function handleToggleCompare(id: string) {
        setCompareIds(prev => {
            if (prev.includes(id)) {
                return prev.filter(x => x !== id);
            }
            if (prev.length >= 2) {
                // Replace the oldest selection
                return [prev[1], id];
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

    return (
        <div className={styles.experiments}>
            <div className={styles.chartSection}>
                <MetricChart experiments={exps} metricKey="brier" />
            </div>
            <div className={styles.tableSection}>
                <ExperimentTable
                    experiments={exps}
                    selectedId={selectedId}
                    onSelect={handleSelect}
                    selectedIds={compareIds}
                    onToggleCompare={handleToggleCompare}
                />
            </div>
            {compareIds.length === 2 && (
                <ComparePanel experiments={exps} selectedIds={compareIds} />
            )}
        </div>
    );
}
