import { NavLink, Link, Outlet, useOutletContext } from 'react-router-dom';
import { useWebSocket, type Event } from '../../hooks/useWebSocket';
import { useApi } from '../../hooks/useApi';
import styles from './Layout.module.css';

const TABS = [
    { path: '/activity', label: 'Activity' },
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

// ASCII art from README rendered as SVG pixel grid
const ASCII_ART = [
    '██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗    ███╗   ███╗██╗     ',
    '██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝    ████╗ ████║██║     ',
    '███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗    ██╔████╔██║██║     ',
    '██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██║     ',
    '██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║    ██║ ╚═╝ ██║███████╗',
    '╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝',
];

const FILLED = new Set(['█']);

function buildLogoPixels(): boolean[][] {
    return ASCII_ART.map(line =>
        [...line].map(ch => FILLED.has(ch))
    );
}

const LOGO_PIXELS = buildLogoPixels();
const LOGO_ROWS = LOGO_PIXELS.length;
const LOGO_COLS = Math.max(...LOGO_PIXELS.map(r => r.length));

function AsciiLogo() {
    const cellSize = 2;
    const gap = 0.5;
    const step = cellSize + gap;
    const width = LOGO_COLS * step;
    const height = LOGO_ROWS * step;

    const rects: React.ReactElement[] = [];
    for (let r = 0; r < LOGO_ROWS; r++) {
        for (let c = 0; c < LOGO_PIXELS[r].length; c++) {
            if (LOGO_PIXELS[r][c]) {
                rects.push(
                    <rect
                        key={`${r}-${c}`}
                        x={c * step}
                        y={r * step}
                        width={cellSize}
                        height={cellSize}
                        rx={0.3}
                    />
                );
            }
        }
    }

    return (
        <svg
            className={styles.logoSvg}
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="HarnessML"
        >
            {rects}
        </svg>
    );
}

export function Layout() {
    const { events, connected } = useWebSocket();
    const { data: status } = useApi<ProjectStatus>('/api/project/status');

    return (
        <div className={styles.layout}>
            <header className={styles.header}>
                <div className={styles.headerLeft}>
                    <Link to="/" className={styles.logoLink}>
                        <span className={styles.logo}>
                            <AsciiLogo />
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
                    </Link>
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
