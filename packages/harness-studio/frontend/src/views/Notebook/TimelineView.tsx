import { useState, useMemo, useRef, useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import { EmptyState } from '../../components/EmptyState/EmptyState';
import { humanizeTimestamp, formatTimestamp } from '../../utils/time';
import type { NotebookEntry, NotebookEntryType } from '../../types/api';
import styles from './Notebook.module.css';

interface TimelineViewProps {
    entries: NotebookEntry[];
    onTagClick: (tag: string) => void;
}

const ENTRY_TYPES: NotebookEntryType[] = ['theory', 'finding', 'research', 'decision', 'plan', 'note'];


function StatsBar({ entries }: { entries: NotebookEntry[] }) {
    const counts = useMemo(() => {
        const map: Record<string, number> = {};
        for (const e of entries) {
            if (!e.struck) {
                map[e.type] = (map[e.type] ?? 0) + 1;
            }
        }
        return map;
    }, [entries]);

    const activeCount = entries.filter(e => !e.struck).length;
    const struckCount = entries.filter(e => e.struck).length;

    const typesWithCounts = ENTRY_TYPES.filter(t => counts[t]);

    return (
        <div className={styles.timelineStats}>
            {typesWithCounts.map((type, idx) => (
                <span key={type} className={styles.statPill}>
                    <TypeBadge type={type} />
                    <span className={styles.statCount}>{counts[type]}</span>
                    {idx < typesWithCounts.length - 1 && <span className={styles.statDivider} />}
                </span>
            ))}
            <span className={styles.statTotal}>
                {activeCount} active{struckCount > 0 ? ` / ${struckCount} struck` : ''}
            </span>
        </div>
    );
}

function DetailPanel({ entry, onTagClick }: { entry: NotebookEntry; onTagClick: (tag: string) => void }) {
    const { project } = useParams<{ project: string }>();
    const allTags = [...entry.tags, ...entry.auto_tags];

    return (
        <div className={styles.timelineDetail}>
            <div className={styles.detailHeader}>
                <TypeBadge type={entry.type} />
                <span className={styles.detailMetaItem}>
                    <strong>{humanizeTimestamp(entry.timestamp)}</strong>
                    {' '}&middot;{' '}
                    {formatTimestamp(entry.timestamp)}
                </span>
                {entry.experiment_id && (
                    <Link
                        to={`/${project}/experiments`}
                        state={{ expandedId: entry.experiment_id }}
                        className={styles.detailMetaItem}
                        style={{ color: 'var(--color-accent)' }}
                    >
                        {entry.experiment_id}
                    </Link>
                )}
                {entry.struck && entry.struck_reason && (
                    <span className={styles.detailMetaItem} style={{ color: 'var(--color-warning)' }}>
                        struck: {entry.struck_reason}
                    </span>
                )}
            </div>
            <div className={styles.detailContent}>
                <MarkdownRenderer content={entry.content} />
            </div>
            {allTags.length > 0 && (
                <div className={styles.detailMeta}>
                    {allTags.map(tag => (
                        <span
                            key={tag}
                            className={styles.detailMetaItem}
                            style={{ cursor: 'pointer', color: 'var(--color-accent)' }}
                            onClick={() => onTagClick(tag)}
                        >
                            #{tag}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}

export function TimelineView({ entries, onTagClick }: TimelineViewProps) {
    const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Reverse so oldest is on the left, newest on the right
    const chronological = useMemo(() => [...entries].reverse(), [entries]);
    const lastIdx = chronological.length - 1;

    // Default to latest entry (rightmost)
    const latestId = chronological.length > 0 ? chronological[lastIdx].id : null;
    const activeId = selectedEntryId ?? latestId;

    const selectedEntry = useMemo(
        () => activeId ? chronological.find(e => e.id === activeId) ?? null : null,
        [activeId, chronological]
    );

    // Auto-scroll to the right (newest) on mount
    useEffect(() => {
        const el = scrollRef.current;
        if (el) {
            el.scrollLeft = el.scrollWidth;
        }
    }, [chronological.length]);

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

    return (
        <div className={styles.timelineWrap}>
            <StatsBar entries={entries} />

            {/* Direction hint */}
            <div className={styles.timelineDirection}>
                <span className={styles.timelineDirectionLabel}>oldest</span>
                <span className={styles.timelineDirectionLine} />
                <span className={styles.timelineDirectionLabel}>→</span>
                <span className={styles.timelineDirectionLine} />
                <span className={styles.timelineDirectionLabel}>newest</span>
            </div>

            <div className={styles.timeline} ref={scrollRef}>
                {chronological.map((entry, idx) => {
                    const isSelected = entry.id === activeId;
                    const isLast = idx === lastIdx;
                    const cardClass = [
                        styles.timelineCardCompact,
                        isSelected && styles.timelineCardCompactSelected,
                        entry.struck && styles.timelineCardCompactStruck,
                    ].filter(Boolean).join(' ');
                    const colClass = styles.timelineColumn;

                    return (
                        <div key={entry.id} className={colClass}>
                            {/* Card */}
                            <div className={styles.timelineCardSlot}>
                                <div
                                    className={cardClass}
                                    onClick={() => setSelectedEntryId(entry.id)}
                                >
                                    <div className={styles.timelineCardCompactMeta}>
                                        <TypeBadge type={entry.type} />
                                        <span className={styles.timelineCardCompactTime}>
                                            {humanizeTimestamp(entry.timestamp)}
                                        </span>
                                    </div>
                                    <div className={styles.timelineCardCompactBody}>
                                        <MarkdownRenderer content={entry.content} />
                                    </div>
                                </div>
                            </div>

                            {/* Selected/Latest label between card and dot */}
                            <div className={styles.timelineLabelSlot}>
                                {isSelected && (
                                    <div className={styles.timelineSelectedLabel}>
                                        {isLast ? 'Latest' : 'Selected'}
                                    </div>
                                )}
                            </div>

                            {/* Dot */}
                            <div className={styles.timelineDotSlot}>
                                <div className={`${styles.timelineDot} ${isLast ? styles.timelineDotLatest : ''}`} />
                            </div>

                            {/* Date + connector */}
                            <div className={styles.timelineDateSlot}>
                                <div className={styles.timelineDate}>
                                    {humanizeTimestamp(entry.timestamp)}
                                </div>
                                {isSelected && (
                                    <div className={styles.timelineConnectorLine} />
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Detail panel — always visible */}
            {selectedEntry && (
                <DetailPanel
                    entry={selectedEntry}
                    onTagClick={onTagClick}
                />
            )}
        </div>
    );
}
