import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useProject } from '../../hooks/useProject';
import { TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import type { NotebookEntry, Experiment } from '../../types/api';
import { StatBar } from './StatBar';
import { EventLog } from './EventLog';
import styles from './Activity.module.css';

function ContextBar({ entries, experiments }: { entries: NotebookEntry[]; experiments: Experiment[] }) {
    const { project } = useParams<{ project: string }>();
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);
    const latestTheory = nonStruck.find(e => e.type === 'theory') ?? null;
    const latestPlan = nonStruck.find(e => e.type === 'plan') ?? null;
    const latestExperiment = experiments.length > 0 ? experiments[0] : null;

    if (!latestTheory && !latestPlan && !latestExperiment) return null;

    return (
        <div className={styles.contextBar}>
            {latestTheory && (
                <div className={styles.contextItem}>
                    <TypeBadge type="theory" />
                    <div className={styles.contextText}>
                        <MarkdownRenderer content={latestTheory.content} />
                    </div>
                </div>
            )}
            {latestPlan && (
                <div className={styles.contextItem}>
                    <TypeBadge type="plan" />
                    <div className={styles.contextText}>
                        <MarkdownRenderer content={latestPlan.content} />
                    </div>
                </div>
            )}
            {latestExperiment && (
                <div className={styles.contextItem}>
                    <span className={styles.contextExpBadge}>{latestExperiment.experiment_id}</span>
                    <div className={styles.contextText}>
                        {latestExperiment.hypothesis ?? latestExperiment.description ?? '--'}
                    </div>
                    {latestExperiment.verdict && (
                        <span className={styles.contextVerdict}>{latestExperiment.verdict}</span>
                    )}
                </div>
            )}
            <Link to={`/${project}/notebook`} className={styles.contextLink}>
                Notebook
            </Link>
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
            <ContextBar entries={notebookEntries ?? []} experiments={experiments ?? []} />
            <StatBar />
            <EventLog wsEvents={events} />
        </div>
    );
}
