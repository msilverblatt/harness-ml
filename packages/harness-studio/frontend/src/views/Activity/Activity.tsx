import { useMemo } from 'react';
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

function ContextCards({ entries, experiments }: { entries: NotebookEntry[]; experiments: Experiment[] }) {
    const { project } = useParams<{ project: string }>();
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);
    const latestTheory = nonStruck.find(e => e.type === 'theory') ?? null;
    const latestPlan = nonStruck.find(e => e.type === 'plan') ?? null;
    const latestExperiment = experiments.length > 0 ? experiments[0] : null;

    return (
        <div className={styles.contextSection}>
            <div className={styles.cardRow}>
                <div className={styles.narrativeCard}>
                    <div className={styles.narrativeLabel}>
                        <TypeBadge type="theory" /> Current Theory
                    </div>
                    <div className={styles.narrativeContent}>
                        {latestTheory ? stripMarkdown(latestTheory.content) : (
                            <span className={styles.narrativeEmpty}>No theory logged yet</span>
                        )}
                    </div>
                </div>
                <div className={styles.narrativeCard}>
                    <div className={styles.narrativeLabel}>
                        <TypeBadge type="plan" /> Current Plan
                    </div>
                    <div className={styles.narrativeContent}>
                        {latestPlan ? stripMarkdown(latestPlan.content) : (
                            <span className={styles.narrativeEmpty}>No plan logged yet</span>
                        )}
                    </div>
                </div>
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
