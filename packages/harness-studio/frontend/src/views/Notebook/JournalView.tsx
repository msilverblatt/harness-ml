import { useMemo } from 'react';
import { NotebookEntryCard, TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import { EmptyState } from '../../components/EmptyState/EmptyState';
import type { NotebookEntry } from '../../types/api';
import styles from './Notebook.module.css';

interface JournalViewProps {
    entries: NotebookEntry[];
    onTagClick: (tag: string) => void;
}

interface ExperimentGroup {
    label: string;
    entries: NotebookEntry[];
    latestTimestamp: string;
}

export function JournalView({ entries, onTagClick }: JournalViewProps) {
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);

    const latestTheory = useMemo(
        () => nonStruck.find(e => e.type === 'theory') ?? null,
        [nonStruck],
    );

    const latestPlan = useMemo(
        () => nonStruck.find(e => e.type === 'plan') ?? null,
        [nonStruck],
    );

    const heroIds = useMemo(() => {
        const ids = new Set<string>();
        if (latestTheory) ids.add(latestTheory.id);
        if (latestPlan) ids.add(latestPlan.id);
        return ids;
    }, [latestTheory, latestPlan]);

    const experimentGroups = useMemo(() => {
        const remaining = entries.filter(e => !heroIds.has(e.id));
        const groupMap = new Map<string, NotebookEntry[]>();

        for (const entry of remaining) {
            const key = entry.experiment_id ?? '__general__';
            const list = groupMap.get(key);
            if (list) {
                list.push(entry);
            } else {
                groupMap.set(key, [entry]);
            }
        }

        const groups: ExperimentGroup[] = [];
        for (const [key, groupEntries] of groupMap) {
            groups.push({
                label: key === '__general__' ? 'General' : key,
                entries: groupEntries,
                latestTimestamp: groupEntries[0]?.timestamp ?? '',
            });
        }

        // Sort by most recent entry, General last
        groups.sort((a, b) => {
            if (a.label === 'General') return 1;
            if (b.label === 'General') return -1;
            return b.latestTimestamp.localeCompare(a.latestTimestamp);
        });

        return groups;
    }, [entries, heroIds]);

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

    const hasHero = latestTheory || latestPlan;

    return (
        <div className={styles.journalSection}>
            {hasHero && (
                <div className={styles.heroCards}>
                    {latestTheory && (
                        <div className={styles.heroCard}>
                            <div className={styles.heroCardLabel}>
                                <TypeBadge type="theory" />
                                Current Theory
                            </div>
                            <div className={styles.heroCardContent}>
                                <MarkdownRenderer content={latestTheory.content} />
                            </div>
                        </div>
                    )}
                    {latestPlan && (
                        <div className={styles.heroCard}>
                            <div className={styles.heroCardLabel}>
                                <TypeBadge type="plan" />
                                Current Plan
                            </div>
                            <div className={styles.heroCardContent}>
                                <MarkdownRenderer content={latestPlan.content} />
                            </div>
                        </div>
                    )}
                </div>
            )}

            {experimentGroups.map(group => (
                <div key={group.label} className={styles.experimentGroup}>
                    <div className={styles.experimentGroupHeader}>{group.label}</div>
                    {group.entries.map(entry => (
                        <NotebookEntryCard
                            key={entry.id}
                            entry={entry}
                            onTagClick={onTagClick}
                        />
                    ))}
                </div>
            ))}
        </div>
    );
}
