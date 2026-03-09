import { Link, useParams } from 'react-router-dom';
import { MarkdownRenderer } from '../MarkdownRenderer/MarkdownRenderer';
import { humanizeTimestamp, formatTimestamp } from '../../utils/time';
import type { NotebookEntry as NotebookEntryType, NotebookEntryType as EntryTypeEnum } from '../../types/api';
import styles from './NotebookEntry.module.css';

const TYPE_COLORS: Record<EntryTypeEnum, { bg: string; fg: string }> = {
    theory: { bg: 'var(--color-nb-theory-a15)', fg: 'var(--color-nb-theory)' },
    finding: { bg: 'var(--color-nb-finding-a15)', fg: 'var(--color-nb-finding)' },
    research: { bg: 'var(--color-nb-research-a15)', fg: 'var(--color-nb-research)' },
    decision: { bg: 'var(--color-nb-decision-a15)', fg: 'var(--color-nb-decision)' },
    plan: { bg: 'var(--color-nb-plan-a15)', fg: 'var(--color-nb-plan)' },
    note: { bg: 'var(--color-nb-note-a15)', fg: 'var(--color-nb-note)' },
};

interface NotebookEntryCardProps {
    entry: NotebookEntryType;
    compact?: boolean;
    onTagClick?: (tag: string) => void;
}

export function TypeBadge({ type }: { type: EntryTypeEnum }) {
    const color = TYPE_COLORS[type] ?? TYPE_COLORS.note;
    return (
        <span
            className={styles.typeBadge}
            style={{ backgroundColor: color.bg, color: color.fg }}
        >
            {type}
        </span>
    );
}

export function NotebookEntryCard({ entry, compact, onTagClick }: NotebookEntryCardProps) {
    const { project } = useParams<{ project: string }>();
    const allTags = [...entry.tags, ...entry.auto_tags];
    const cardClass = [
        styles.entryCard,
        compact && styles.entryCompact,
        entry.struck && styles.entryCardStruck,
    ].filter(Boolean).join(' ');

    const borderColor = TYPE_COLORS[entry.type]?.fg ?? 'var(--color-border)';

    return (
        <div className={cardClass} style={compact ? { borderLeftColor: borderColor } : undefined}>
            <div className={styles.entryMeta}>
                <TypeBadge type={entry.type} />
                <span className={styles.timestamp} title={formatTimestamp(entry.timestamp)}>
                    {humanizeTimestamp(entry.timestamp)}
                </span>
                {entry.experiment_id && (
                    <Link
                        to={`/${project}/experiments`}
                        state={{ expandedId: entry.experiment_id }}
                        className={styles.experimentLink}
                    >
                        {entry.experiment_id}
                    </Link>
                )}
                {entry.struck && entry.struck_reason && (
                    <span className={styles.struckReason}>struck: {entry.struck_reason}</span>
                )}
            </div>
            <div className={styles.entryContent}>
                <MarkdownRenderer content={entry.content} />
            </div>
            {allTags.length > 0 && !compact && (
                <div className={styles.tagList}>
                    {allTags.map(tag => (
                        <span
                            key={tag}
                            className={styles.tag}
                            onClick={() => onTagClick?.(tag)}
                        >
                            {tag}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}

interface StruckIndicatorProps {
    count: number;
    expanded: boolean;
    onToggle: () => void;
}

export function StruckIndicator({ count, expanded, onToggle }: StruckIndicatorProps) {
    if (count === 0) return null;
    return (
        <div className={styles.struckIndicator} onClick={onToggle}>
            {expanded ? 'Hide' : 'Show'} {count} previous {count === 1 ? 'version' : 'versions'}
        </div>
    );
}
