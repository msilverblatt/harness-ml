import { useMemo } from 'react';
import styles from './Diagnostics.module.css';

interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    has_report?: boolean;
    experiment_id?: string | null;
}

interface RunSelectorProps {
    runs: RunSummary[];
    selectedRunIds: string[];
    onSelect: (runIds: string[]) => void;
    compareMode: boolean;
    onToggleCompare: () => void;
}

const LOWER_IS_BETTER: Record<string, boolean> = {
    brier: true,
    rmse: true,
    mae: true,
    mape: true,
    log_loss: true,
    ece: true,
};

function formatTimestamp(id: string): string {
    const match = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (!match) return id;
    const [, year, month, day, hour, minute] = match;
    const date = new Date(
        parseInt(year), parseInt(month) - 1, parseInt(day),
        parseInt(hour), parseInt(minute)
    );
    return date.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
    }) + ' ' + date.toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', hour12: true,
    });
}

function getPrimaryMetric(metrics: Record<string, number>): { name: string; value: number } | null {
    const priority = ['brier', 'rmse', 'accuracy', 'r_squared', 'mae', 'log_loss', 'ece'];
    for (const key of priority) {
        if (metrics[key] !== undefined) {
            return { name: key, value: metrics[key] };
        }
    }
    const keys = Object.keys(metrics);
    if (keys.length > 0) {
        return { name: keys[0], value: metrics[keys[0]] };
    }
    return null;
}

function findBestRunId(runs: RunSummary[]): string | null {
    const runsWithMetrics = runs.filter(r => Object.keys(r.metrics).length > 0);
    if (runsWithMetrics.length === 0) return null;

    const primary = getPrimaryMetric(runsWithMetrics[0].metrics);
    if (!primary) return null;

    const lowerBetter = LOWER_IS_BETTER[primary.name] ?? false;
    let best = runsWithMetrics[0];
    for (const run of runsWithMetrics.slice(1)) {
        const val = run.metrics[primary.name];
        if (val === undefined) continue;
        const bestVal = best.metrics[primary.name];
        if (lowerBetter ? val < bestVal : val > bestVal) {
            best = run;
        }
    }
    return best.id;
}

export function RunSelector({ runs, selectedRunIds, onSelect, compareMode, onToggleCompare }: RunSelectorProps) {
    const bestRunId = useMemo(() => findBestRunId(runs), [runs]);
    const latestRunId = runs.length > 0 ? runs[0].id : null;

    const handleRunClick = (runId: string) => {
        if (compareMode) {
            if (selectedRunIds.includes(runId)) {
                const next = selectedRunIds.filter(id => id !== runId);
                if (next.length > 0) onSelect(next);
            } else if (selectedRunIds.length < 3) {
                onSelect([...selectedRunIds, runId]);
            }
        } else {
            onSelect([runId]);
        }
    };

    return (
        <div className={styles.runSelector}>
            <div className={styles.runSelectorHeader}>
                <span className={styles.selectorLabel}>Runs</span>
                <div className={styles.runQuickActions}>
                    <button
                        className={`${styles.quickBtn} ${latestRunId && selectedRunIds.includes(latestRunId) ? styles.quickBtnActive : ''}`}
                        onClick={() => latestRunId && onSelect([latestRunId])}
                        disabled={!latestRunId}
                    >
                        Latest
                    </button>
                    <button
                        className={`${styles.quickBtn} ${bestRunId && selectedRunIds.includes(bestRunId) ? styles.quickBtnActive : ''}`}
                        onClick={() => bestRunId && onSelect([bestRunId])}
                        disabled={!bestRunId}
                    >
                        Best
                    </button>
                    <button
                        className={`${styles.quickBtn} ${compareMode ? styles.quickBtnActive : ''}`}
                        onClick={onToggleCompare}
                    >
                        Compare
                    </button>
                </div>
            </div>
            <div className={styles.runList}>
                {runs.map(run => {
                    const isSelected = selectedRunIds.includes(run.id);
                    const primary = getPrimaryMetric(run.metrics);
                    const isBest = run.id === bestRunId;
                    const hasMetrics = Object.keys(run.metrics).length > 0;
                    return (
                        <div
                            key={run.id}
                            className={`${styles.runItem} ${isSelected ? styles.runItemSelected : ''}`}
                            onClick={() => handleRunClick(run.id)}
                        >
                            {compareMode && (
                                <input
                                    type="checkbox"
                                    checked={isSelected}
                                    readOnly
                                    className={styles.runCheckbox}
                                />
                            )}
                            <div className={styles.runItemContent}>
                                <span className={styles.runTimestamp}>
                                    {formatTimestamp(run.id)}
                                </span>
                                <div className={styles.runBadges}>
                                    {run.experiment_id && (
                                        <span className={styles.expBadge}>{run.experiment_id}</span>
                                    )}
                                    {isBest && hasMetrics && (
                                        <span className={styles.bestBadge}>best</span>
                                    )}
                                </div>
                            </div>
                            {primary && (
                                <span className={styles.runMetricInline}>
                                    {primary.name}: {primary.value.toFixed(4)}
                                </span>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
