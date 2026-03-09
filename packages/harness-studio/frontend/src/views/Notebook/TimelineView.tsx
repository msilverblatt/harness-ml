import { useState, useMemo } from 'react';
import { NotebookEntryCard, StruckIndicator } from '../../components/NotebookEntry/NotebookEntry';
import { EmptyState } from '../../components/EmptyState/EmptyState';
import type { NotebookEntry } from '../../types/api';
import styles from './Notebook.module.css';

interface TimelineViewProps {
    entries: NotebookEntry[];
    onTagClick: (tag: string) => void;
}

interface EntryGroup {
    visible: NotebookEntry;
    struck: NotebookEntry[];
}

/**
 * Group entries so that consecutive struck entries of the same type
 * that follow a non-struck entry are collapsed behind that entry.
 */
function groupEntries(entries: NotebookEntry[]): EntryGroup[] {
    const groups: EntryGroup[] = [];

    for (const entry of entries) {
        if (!entry.struck) {
            groups.push({ visible: entry, struck: [] });
        } else {
            // Attach to the previous group if same type and previous is non-struck
            const prev = groups[groups.length - 1];
            if (prev && prev.visible.type === entry.type && !prev.visible.struck) {
                prev.struck.push(entry);
            } else {
                // Standalone struck entry (no non-struck predecessor of same type)
                groups.push({ visible: entry, struck: [] });
            }
        }
    }

    return groups;
}

export function TimelineView({ entries, onTagClick }: TimelineViewProps) {
    const [expandedIndices, setExpandedIndices] = useState<Set<number>>(new Set());
    const groups = useMemo(() => groupEntries(entries), [entries]);

    if (entries.length === 0) {
        return (
            <div className={styles.emptyState}>
                <EmptyState
                    title="No entries match"
                    description="Adjust your filters or add notebook entries via the MCP notebook tool."
                />
            </div>
        );
    }

    function toggleExpanded(idx: number) {
        setExpandedIndices(prev => {
            const next = new Set(prev);
            if (next.has(idx)) {
                next.delete(idx);
            } else {
                next.add(idx);
            }
            return next;
        });
    }

    return (
        <div className={styles.timeline}>
            {groups.map((group, idx) => (
                <div key={group.visible.id}>
                    <NotebookEntryCard entry={group.visible} onTagClick={onTagClick} />
                    {group.struck.length > 0 && (
                        <>
                            <StruckIndicator
                                count={group.struck.length}
                                expanded={expandedIndices.has(idx)}
                                onToggle={() => toggleExpanded(idx)}
                            />
                            {expandedIndices.has(idx) && group.struck.map(s => (
                                <NotebookEntryCard
                                    key={s.id}
                                    entry={s}
                                    compact
                                    onTagClick={onTagClick}
                                />
                            ))}
                        </>
                    )}
                </div>
            ))}
        </div>
    );
}
