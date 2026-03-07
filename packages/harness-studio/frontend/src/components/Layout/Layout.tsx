import { NavLink, Outlet, useOutletContext } from 'react-router-dom';
import { useWebSocket, Event } from '../../hooks/useWebSocket';
import { useApi } from '../../hooks/useApi';
import styles from './Layout.module.css';

const TABS = [
    { path: '/', label: 'Activity' },
    { path: '/dag', label: 'DAG' },
    { path: '/experiments', label: 'Experiments' },
    { path: '/diagnostics', label: 'Diagnostics' },
];

interface ProjectStatus {
    project_name: string;
    task: string;
    active_models: number;
    experiments_run: number;
    run_count: number;
    feature_count: number;
    latest_metrics: Record<string, number>;
}

export interface LayoutContext {
    events: Event[];
    connected: boolean;
}

export function useLayoutContext() {
    return useOutletContext<LayoutContext>();
}

export function Layout() {
    const { events, connected } = useWebSocket();
    const { data: status } = useApi<ProjectStatus>('/api/project/status');

    return (
        <div className={styles.layout}>
            <header className={styles.header}>
                <div className={styles.headerLeft}>
                    <span className={styles.logo}>
                        <span className={styles.logoName}>HarnessML</span>
                        <span className={styles.logoBadge}>Studio</span>
                    </span>
                    {status?.project_name && (
                        <>
                            <span className={styles.divider} />
                            <span className={styles.project}>
                                <span className={styles.projectName}>{status.project_name}</span>
                                {status.task && (
                                    <span className={styles.projectTask}>{status.task}</span>
                                )}
                            </span>
                        </>
                    )}
                </div>
                <nav className={styles.tabs}>
                    {TABS.map(t => (
                        <NavLink key={t.path} to={t.path} end
                            className={({ isActive }) =>
                                isActive ? styles.tabActive : styles.tab
                            }>
                            {t.label}
                        </NavLink>
                    ))}
                </nav>
                <div className={styles.headerRight}>
                    <span className={connected ? styles.statusLive : styles.statusOffline}>
                        <span className={styles.statusDot} />
                        {connected ? 'Live' : 'Offline'}
                    </span>
                </div>
            </header>
            <main className={styles.content}>
                <Outlet context={{ events, connected } satisfies LayoutContext} />
            </main>
        </div>
    );
}
