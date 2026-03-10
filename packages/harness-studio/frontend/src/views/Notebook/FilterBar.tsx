import { useMemo } from 'react';
import type { NotebookEntry, NotebookEntryType } from '../../types/api';
import styles from './Notebook.module.css';

export interface NotebookFilters {
    type: string;
    text: string;
    tag: string;
    experimentId: string;
    dateFrom: string;
    dateTo: string;
    showStruck: boolean;
}

export const EMPTY_FILTERS: NotebookFilters = {
    type: '',
    text: '',
    tag: '',
    experimentId: '',
    dateFrom: '',
    dateTo: '',
    showStruck: false,
};

const ENTRY_TYPES: NotebookEntryType[] = ['theory', 'finding', 'research', 'decision', 'plan', 'note'];

export function filterEntries(entries: NotebookEntry[], filters: NotebookFilters): NotebookEntry[] {
    return entries.filter(entry => {
        if (!filters.showStruck && entry.struck) return false;
        if (filters.type && entry.type !== filters.type) return false;
        if (filters.text) {
            const needle = filters.text.toLowerCase();
            if (!entry.content.toLowerCase().includes(needle)) return false;
        }
        if (filters.tag) {
            const allTags = [...entry.tags, ...entry.auto_tags];
            if (!allTags.includes(filters.tag)) return false;
        }
        if (filters.experimentId) {
            if (entry.experiment_id !== filters.experimentId) return false;
        }
        if (filters.dateFrom) {
            if (entry.timestamp < filters.dateFrom) return false;
        }
        if (filters.dateTo) {
            if (entry.timestamp > filters.dateTo + 'T23:59:59') return false;
        }
        return true;
    });
}

function filtersActive(filters: NotebookFilters): boolean {
    return (
        filters.type !== '' ||
        filters.text !== '' ||
        filters.tag !== '' ||
        filters.experimentId !== '' ||
        filters.dateFrom !== '' ||
        filters.dateTo !== '' ||
        filters.showStruck
    );
}

interface FilterBarProps {
    filters: NotebookFilters;
    onChange: (filters: NotebookFilters) => void;
    entries: NotebookEntry[];
    filteredCount: number;
}

export function FilterBar({ filters, onChange, entries, filteredCount }: FilterBarProps) {
    const allTags = useMemo(() => {
        const tagSet = new Set<string>();
        for (const e of entries) {
            for (const t of e.tags) tagSet.add(t);
            for (const t of e.auto_tags) tagSet.add(t);
        }
        return Array.from(tagSet).sort();
    }, [entries]);

    const experimentIds = useMemo(() => {
        const ids = new Set<string>();
        for (const e of entries) {
            if (e.experiment_id) ids.add(e.experiment_id);
        }
        return Array.from(ids).sort();
    }, [entries]);

    const struckCount = useMemo(() => entries.filter(e => e.struck).length, [entries]);

    function set(patch: Partial<NotebookFilters>) {
        onChange({ ...filters, ...patch });
    }

    return (
        <div className={styles.filterBar}>
            <select
                className={styles.filterSelect}
                value={filters.type}
                onChange={e => set({ type: e.target.value })}
            >
                <option value="">All types</option>
                {ENTRY_TYPES.map(t => (
                    <option key={t} value={t}>{t}</option>
                ))}
            </select>

            <input
                className={styles.filterInput}
                type="text"
                placeholder="Search..."
                value={filters.text}
                onChange={e => set({ text: e.target.value })}
            />

            {allTags.length > 0 && (
                <select
                    className={styles.filterSelect}
                    value={filters.tag}
                    onChange={e => set({ tag: e.target.value })}
                >
                    <option value="">All tags</option>
                    {allTags.map(t => (
                        <option key={t} value={t}>{t}</option>
                    ))}
                </select>
            )}

            {experimentIds.length > 0 && (
                <select
                    className={styles.filterSelect}
                    value={filters.experimentId}
                    onChange={e => set({ experimentId: e.target.value })}
                >
                    <option value="">All experiments</option>
                    {experimentIds.map(id => (
                        <option key={id} value={id}>{id}</option>
                    ))}
                </select>
            )}

            <input
                className={styles.dateInput}
                type="date"
                value={filters.dateFrom}
                onChange={e => set({ dateFrom: e.target.value })}
                title="From date"
            />
            <input
                className={styles.dateInput}
                type="date"
                value={filters.dateTo}
                onChange={e => set({ dateTo: e.target.value })}
                title="To date"
            />

            {struckCount > 0 && (
                <button
                    className={filters.showStruck ? styles.struckToggleActive : styles.struckToggle}
                    onClick={() => set({ showStruck: !filters.showStruck })}
                >
                    struck ({struckCount})
                </button>
            )}

            {filtersActive(filters) && (
                <button
                    className={styles.clearButton}
                    onClick={() => onChange(EMPTY_FILTERS)}
                >
                    clear
                </button>
            )}

            <span className={styles.filterCount}>
                {filteredCount} / {entries.length}
            </span>
        </div>
    );
}
