import { useState, useMemo, useRef, useEffect } from 'react';
import { MetricLabel } from '../../components/Tooltip/Tooltip';
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

type FilterType = 'all' | 'baseline' | 'experiment';

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
        month: 'short', day: 'numeric',
    }) + ' ' + date.toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', hour12: true,
    });
}

function getRunLabel(run: RunSummary): string {
    if (run.experiment_id) return run.experiment_id;
    return 'Baseline';
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
    const [filter, setFilter] = useState<FilterType>('all');
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const bestRunId = useMemo(() => findBestRunId(runs), [runs]);

    useEffect(() => {
        function handleClickOutside(e: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
                setDropdownOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const filteredRuns = useMemo(() => {
        if (filter === 'baseline') return runs.filter(r => !r.experiment_id);
        if (filter === 'experiment') return runs.filter(r => !!r.experiment_id);
        return runs;
    }, [runs, filter]);

    const primaryRun = runs.find(r => r.id === selectedRunIds[0]);

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
            setDropdownOpen(false);
        }
    };

    return (
        <div className={styles.selectorBar} ref={dropdownRef}>
            <div className={styles.selectorLeft}>
                <button
                    className={styles.selectorTrigger}
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                >
                    <span className={styles.selectorRunLabel}>
                        {primaryRun ? (
                            <>
                                <span className={primaryRun.experiment_id ? styles.tagExperiment : styles.tagBaseline}>
                                    {getRunLabel(primaryRun)}
                                </span>
                                <span className={styles.selectorTimestamp}>{formatTimestamp(primaryRun.id)}</span>
                                {primaryRun.id === bestRunId && (
                                    <span className={styles.tagBest}>best</span>
                                )}
                            </>
                        ) : (
                            'Select a run...'
                        )}
                    </span>
                    <span className={styles.selectorChevron}>{dropdownOpen ? '\u25B2' : '\u25BC'}</span>
                </button>

                <div className={styles.filterChips}>
                    {(['all', 'baseline', 'experiment'] as FilterType[]).map(f => (
                        <button
                            key={f}
                            className={`${styles.filterChip} ${filter === f ? styles.filterChipActive : ''}`}
                            onClick={() => setFilter(f)}
                        >
                            {f === 'all' ? 'All' : f === 'baseline' ? 'Baseline' : 'Experiments'}
                        </button>
                    ))}
                </div>
            </div>

            <div className={styles.selectorRight}>
                <button
                    className={`${styles.compareBtn} ${compareMode ? styles.compareBtnActive : ''}`}
                    onClick={onToggleCompare}
                >
                    Compare{compareMode && selectedRunIds.length > 1 ? ` (${selectedRunIds.length})` : ''}
                </button>
            </div>

            {dropdownOpen && (
                <div className={styles.dropdown}>
                    {filteredRuns.length === 0 && (
                        <div className={styles.dropdownEmpty}>No matching runs</div>
                    )}
                    {filteredRuns.map(run => {
                        const isSelected = selectedRunIds.includes(run.id);
                        const primary = getPrimaryMetric(run.metrics);
                        const isBest = run.id === bestRunId;
                        return (
                            <div
                                key={run.id}
                                className={`${styles.dropdownItem} ${isSelected ? styles.dropdownItemSelected : ''}`}
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
                                <span className={run.experiment_id ? styles.tagExperiment : styles.tagBaseline}>
                                    {getRunLabel(run)}
                                </span>
                                <span className={styles.dropdownTimestamp}>
                                    {formatTimestamp(run.id)}
                                </span>
                                {isBest && <span className={styles.tagBest}>best</span>}
                                {primary && (
                                    <span className={styles.dropdownMetric}>
                                        <MetricLabel name={primary.name} />: {primary.value.toFixed(4)}
                                    </span>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
