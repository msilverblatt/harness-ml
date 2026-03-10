import { useState, useMemo } from 'react';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useProject } from '../../hooks/useProject';
import { EmptyState } from '../../components/EmptyState/EmptyState';
import { FilterBar, filterEntries, EMPTY_FILTERS } from './FilterBar';
import { TimelineView } from './TimelineView';
import { JournalView } from './JournalView';
import type { NotebookEntry } from '../../types/api';
import type { NotebookFilters } from './FilterBar';
import styles from './Notebook.module.css';

type ViewMode = 'journal' | 'timeline';

export function Notebook() {
    const { events } = useLayoutContext();
    const project = useProject();
    const refreshKey = useRefreshKey(events, ['notebook']);
    const { data: entries, loading, error } = useApi<NotebookEntry[]>('/api/notebook', refreshKey, project);
    const [view, setView] = useState<ViewMode>('journal');
    const [filters, setFilters] = useState<NotebookFilters>(EMPTY_FILTERS);

    const allEntries = entries ?? [];
    const filtered = useMemo(() => filterEntries(allEntries, filters), [allEntries, filters]);

    function handleTagClick(tag: string) {
        setFilters(prev => ({ ...prev, tag }));
    }

    if (loading) {
        return (
            <div className={styles.notebook}>
                <div className={styles.emptyState}>Loading notebook...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.notebook}>
                <div className={styles.emptyState}>Error: {error}</div>
            </div>
        );
    }

    if (allEntries.length === 0) {
        return (
            <div className={styles.notebook}>
                <EmptyState
                    title="No notebook entries"
                    description="Use the MCP notebook tool to record theories, findings, decisions, and plans."
                />
            </div>
        );
    }

    return (
        <div className={styles.notebook}>
            <div className={styles.topBar}>
                <div className={styles.viewToggle}>
                    <button
                        className={view === 'journal' ? styles.viewButtonActive : styles.viewButton}
                        onClick={() => setView('journal')}
                    >
                        journal
                    </button>
                    <button
                        className={view === 'timeline' ? styles.viewButtonActive : styles.viewButton}
                        onClick={() => setView('timeline')}
                    >
                        timeline
                    </button>
                </div>
                <FilterBar
                    filters={filters}
                    onChange={setFilters}
                    entries={allEntries}
                    filteredCount={filtered.length}
                />
            </div>

            {view === 'timeline' ? (
                <TimelineView entries={filtered} onTagClick={handleTagClick} />
            ) : (
                <JournalView entries={filtered} onTagClick={handleTagClick} />
            )}
        </div>
    );
}
