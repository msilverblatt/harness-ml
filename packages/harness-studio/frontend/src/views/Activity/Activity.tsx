import { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useProject } from '../../hooks/useProject';
import { TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import type { NotebookEntry, Experiment } from '../../types/api';
import { EventLog } from './EventLog';
import styles from './Activity.module.css';

/** Strip markdown syntax to plain text, keeping substance. */
function stripMarkdown(content: string): string {
    return content
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 0 && !/^#+\s/.test(l) && !/^---/.test(l))
        .map(l => l
            .replace(/\*\*/g, '')
            .replace(/\*/g, '')
            .replace(/`([^`]+)`/g, '$1')
            .replace(/^\d+\.\s+/, '• ')
            .replace(/^[-*]\s+/, '• ')
        )
        .join('\n');
}

function normalizeVerdict(v: string | undefined): string {
    if (!v) return 'pending';
    const lower = v.toLowerCase();
    if (['keep', 'improved', 'positive'].includes(lower)) return 'positive';
    if (['revert', 'regressed', 'negative', 'failed'].includes(lower)) return 'negative';
    if (['partial', 'neutral', 'baseline'].includes(lower)) return 'neutral';
    return 'pending';
}

function verdictClass(verdict?: string): string {
    const normalized = normalizeVerdict(verdict);
    switch (normalized) {
        case 'positive': return styles.verdictKeep;
        case 'neutral': return styles.verdictPartial;
        case 'negative': return styles.verdictRevert;
        default: return '';
    }
}

function NarrativeCard({ label, badge, content }: { label: string; badge: React.ReactNode; content: string | null }) {
    const [expanded, setExpanded] = useState(false);
    const [overflows, setOverflows] = useState(false);
    const contentRef = useRef<HTMLDivElement>(null);

    const checkOverflow = useCallback(() => {
        const el = contentRef.current;
        if (el) {
            setOverflows(el.scrollHeight > 120);
        }
    }, []);

    useEffect(() => {
        checkOverflow();
    }, [content, checkOverflow]);

    return (
        <div className={styles.narrativeCard}>
            <div className={styles.narrativeLabel}>
                {badge} {label}
            </div>
            <div
                ref={contentRef}
                className={`${styles.narrativeContent}${expanded ? ` ${styles.narrativeContentExpanded}` : ''}`}
            >
                {content ? stripMarkdown(content) : (
                    <span className={styles.narrativeEmpty}>No {label.toLowerCase()} logged yet</span>
                )}
                {!expanded && overflows && <div className={styles.narrativeFade} />}
            </div>
            {overflows && (
                <button className={styles.showMore} onClick={() => setExpanded(e => !e)}>
                    {expanded ? 'Show less' : 'Show more'}
                </button>
            )}
        </div>
    );
}

function ContextCards({ entries, experiments }: { entries: NotebookEntry[]; experiments: Experiment[] }) {
    const { project } = useParams<{ project: string }>();
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);
    const latestTheory = nonStruck.find(e => e.type === 'theory') ?? null;
    const latestPlan = nonStruck.find(e => e.type === 'plan') ?? null;
    const latestExperiment = experiments.length > 0 ? experiments[0] : null;

    return (
        <div className={styles.contextSection}>
            <div className={styles.cardRow}>
                <NarrativeCard
                    label="Current Theory"
                    badge={<TypeBadge type="theory" />}
                    content={latestTheory?.content ?? null}
                />
                <NarrativeCard
                    label="Current Plan"
                    badge={<TypeBadge type="plan" />}
                    content={latestPlan?.content ?? null}
                />
            </div>
            {latestExperiment && (
                <div className={styles.experimentBar}>
                    <span className={styles.expId}>{latestExperiment.experiment_id}</span>
                    <span className={styles.expHypothesis}>
                        {latestExperiment.hypothesis ?? latestExperiment.description ?? '--'}
                    </span>
                    {latestExperiment.verdict && (
                        <span className={`${styles.expVerdict} ${verdictClass(latestExperiment.verdict)}`}>
                            {latestExperiment.verdict}
                        </span>
                    )}
                    <Link to={`/${project}/experiments`} className={styles.expLink}>
                        View all
                    </Link>
                </div>
            )}
        </div>
    );
}

export function Activity() {
    const { events } = useLayoutContext();
    const project = useProject();
    const nbKey = useRefreshKey(events, ['notebook']);
    const expKey = useRefreshKey(events, ['experiments', 'pipeline']);
    const { data: notebookEntries } = useApi<NotebookEntry[]>('/api/notebook', nbKey, project);
    const { data: experiments } = useApi<Experiment[]>('/api/experiments', expKey, project);

    return (
        <div className={styles.activity}>
            <ContextCards entries={notebookEntries ?? []} experiments={experiments ?? []} />
            <EventLog wsEvents={events} />
        </div>
    );
}
