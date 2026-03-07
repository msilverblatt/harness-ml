import { useState } from 'react';
import styles from './Experiments.module.css';

export interface Experiment {
    experiment_id: string;
    timestamp?: string;
    hypothesis?: string;
    changes?: string;
    verdict?: string;
    metrics?: Record<string, number>;
    baseline_metrics?: Record<string, number>;
    primary_delta?: number;
    notes?: string;
}

type SortKey = 'experiment_id' | 'timestamp' | 'hypothesis' | 'verdict' | 'primary_delta' | 'notes';

interface ExperimentTableProps {
    experiments: Experiment[];
    selectedId: string | null;
    onSelect: (id: string) => void;
    selectedIds: string[];
    onToggleCompare: (id: string) => void;
}

function formatDate(ts?: string): string {
    if (!ts) return '--';
    try {
        const d = new Date(ts);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
        return ts;
    }
}

function truncate(s: string | undefined, max: number): string {
    if (!s) return '--';
    return s.length > max ? s.slice(0, max) + '...' : s;
}

function verdictClass(verdict?: string): string {
    switch (verdict) {
        case 'keep': return styles.verdictKeep;
        case 'partial': return styles.verdictPartial;
        case 'revert': return styles.verdictRevert;
        default: return '';
    }
}

function formatDelta(delta?: number): { text: string; className: string } {
    if (delta == null) return { text: '--', className: '' };
    const sign = delta > 0 ? '+' : '';
    const text = `${sign}${delta.toFixed(4)}`;
    // For primary_delta, negative typically means improvement (lower brier/loss)
    const className = delta < 0 ? styles.deltaPositive : delta > 0 ? styles.deltaNegative : '';
    return { text, className };
}

export function ExperimentTable({ experiments, selectedId, onSelect, selectedIds, onToggleCompare }: ExperimentTableProps) {
    const [sortKey, setSortKey] = useState<SortKey>('timestamp');
    const [sortAsc, setSortAsc] = useState(false);

    function handleSort(key: SortKey) {
        if (sortKey === key) {
            setSortAsc(!sortAsc);
        } else {
            setSortKey(key);
            setSortAsc(true);
        }
    }

    function getSortValue(exp: Experiment, key: SortKey): string | number {
        switch (key) {
            case 'experiment_id': return exp.experiment_id;
            case 'timestamp': return exp.timestamp ?? '';
            case 'hypothesis': return exp.hypothesis ?? '';
            case 'verdict': return exp.verdict ?? '';
            case 'primary_delta': return exp.primary_delta ?? 0;
            case 'notes': return exp.notes ?? '';
        }
    }

    const sorted = [...experiments].sort((a, b) => {
        const aVal = getSortValue(a, sortKey);
        const bVal = getSortValue(b, sortKey);
        const cmp = typeof aVal === 'number' && typeof bVal === 'number'
            ? aVal - bVal
            : String(aVal).localeCompare(String(bVal));
        return sortAsc ? cmp : -cmp;
    });

    const sortArrow = (key: SortKey) =>
        sortKey === key ? <span className={styles.sortIndicator}>{sortAsc ? '\u25B2' : '\u25BC'}</span> : null;

    if (experiments.length === 0) {
        return <div className={styles.emptyState}>No experiments recorded yet.</div>;
    }

    return (
        <table className={styles.table}>
            <thead>
                <tr>
                    <th className={styles.th} style={{ width: 32 }}></th>
                    <th className={styles.th} onClick={() => handleSort('experiment_id')}>ID{sortArrow('experiment_id')}</th>
                    <th className={styles.th} onClick={() => handleSort('timestamp')}>Date{sortArrow('timestamp')}</th>
                    <th className={styles.th} onClick={() => handleSort('hypothesis')}>Hypothesis{sortArrow('hypothesis')}</th>
                    <th className={styles.th} onClick={() => handleSort('verdict')}>Verdict{sortArrow('verdict')}</th>
                    <th className={styles.th} onClick={() => handleSort('primary_delta')}>Delta{sortArrow('primary_delta')}</th>
                    <th className={styles.th} onClick={() => handleSort('notes')}>Notes{sortArrow('notes')}</th>
                </tr>
            </thead>
            <tbody>
                {sorted.map(exp => {
                    const isSelected = selectedId === exp.experiment_id;
                    const delta = formatDelta(exp.primary_delta);
                    return (
                        <tr
                            key={exp.experiment_id}
                            className={`${styles.row} ${isSelected ? styles.rowSelected : ''}`}
                            onClick={() => onSelect(exp.experiment_id)}
                        >
                            <td className={styles.td}>
                                <input
                                    type="checkbox"
                                    className={styles.checkbox}
                                    checked={selectedIds.includes(exp.experiment_id)}
                                    onChange={(e) => {
                                        e.stopPropagation();
                                        onToggleCompare(exp.experiment_id);
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                />
                            </td>
                            <td className={styles.td}>{exp.experiment_id}</td>
                            <td className={styles.td}>{formatDate(exp.timestamp)}</td>
                            <td className={styles.td} title={exp.hypothesis}>{truncate(exp.hypothesis, 40)}</td>
                            <td className={styles.td}>
                                <span className={verdictClass(exp.verdict)}>{exp.verdict ?? '--'}</span>
                            </td>
                            <td className={`${styles.td} ${delta.className}`}>{delta.text}</td>
                            <td className={styles.td} title={exp.notes}>{truncate(exp.notes, 40)}</td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}
