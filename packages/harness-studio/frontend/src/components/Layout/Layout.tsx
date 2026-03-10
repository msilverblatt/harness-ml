import { NavLink, Link, Outlet, useOutletContext, useParams, useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState, useRef, useCallback } from 'react';
import { useWebSocket, type Event } from '../../hooks/useWebSocket';
import { useApi } from '../../hooks/useApi';
import { ProjectContext } from '../../hooks/useProject';
import { useToast } from '../Toast/Toast';
import { useKeyboardShortcuts, SHORTCUT_DESCRIPTIONS } from '../../hooks/useKeyboardShortcuts';
import styles from './Layout.module.css';

interface ProjectStatus {
    project_name: string;
    task: string;
    target_column: string | null;
    active_models: number;
    experiments_run: number;
    run_count: number;
    feature_count: number;
    latest_metrics: Record<string, number>;
}

interface ProjectInfo {
    name: string;
    project_dir: string;
    last_seen: number;
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

// Small pixel robot mascot (6 rows x 7 cols)
const ROBOT_PIXELS = [
    [0,1,1,1,1,1,0],
    [1,1,0,1,0,1,1],
    [1,1,1,1,1,1,1],
    [0,1,0,0,0,1,0],
    [0,1,1,1,1,1,0],
    [0,1,0,0,0,1,0],
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

type RobotStatus = 'offline' | 'idle' | 'working';

function AsciiLogo({ robotStatus = 'idle' }: { robotStatus?: RobotStatus }) {
    const cellSize = 2;
    const gap = 0.5;
    const step = cellSize + gap;

    const robotCols = ROBOT_PIXELS[0].length;
    const robotRows = ROBOT_PIXELS.length;
    const robotGap = 2;
    const robotOffsetX = (LOGO_COLS + robotGap) * step;
    const robotOffsetY = ((LOGO_ROWS - robotRows) / 2) * step;

    const extraRight = 6 * step;
    const totalWidth = (LOGO_COLS + robotGap + robotCols) * step + extraRight;
    const height = LOGO_ROWS * step;

    const rects: React.ReactElement[] = [];
    for (let r = 0; r < LOGO_ROWS; r++) {
        for (let c = 0; c < LOGO_PIXELS[r].length; c++) {
            if (LOGO_PIXELS[r][c]) {
                rects.push(
                    <rect
                        key={`t${r}-${c}`}
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

    const robotClass = robotStatus === 'offline'
        ? styles.robotOffline
        : robotStatus === 'working'
            ? styles.robotWorking
            : styles.robotIdle;

    for (let r = 0; r < robotRows; r++) {
        for (let c = 0; c < robotCols; c++) {
            if (ROBOT_PIXELS[r][c]) {
                rects.push(
                    <rect
                        key={`r${r}-${c}`}
                        x={robotOffsetX + c * step}
                        y={robotOffsetY + r * step}
                        width={cellSize}
                        height={cellSize}
                        rx={0.3}
                        className={robotClass}
                    />
                );
            }
        }
    }

    const decoX = robotOffsetX + (robotCols + 1) * step;
    const decoY = robotOffsetY;

    if (robotStatus === 'idle') {
        rects.push(
            <text key="z1" x={decoX} y={decoY + 2} className={styles.zzz1}>z</text>,
            <text key="z2" x={decoX + 3.5} y={decoY - 1.5} className={styles.zzz2}>z</text>,
            <text key="z3" x={decoX + 7} y={decoY - 5} className={styles.zzz3}>z</text>,
        );
    } else if (robotStatus === 'working') {
        const lineX = robotOffsetX - 3 * step;
        rects.push(
            <rect key="m1" x={lineX} y={robotOffsetY + 1 * step} width={2 * step} height={cellSize * 0.5} rx={0.2} className={styles.motionLine1} />,
            <rect key="m2" x={lineX - step} y={robotOffsetY + 3 * step} width={2.5 * step} height={cellSize * 0.5} rx={0.2} className={styles.motionLine2} />,
            <rect key="m3" x={lineX} y={robotOffsetY + 5 * step} width={1.5 * step} height={cellSize * 0.5} rx={0.2} className={styles.motionLine3} />,
        );
    }

    return (
        <svg
            className={styles.logoSvg}
            viewBox={`0 0 ${totalWidth} ${height}`}
            role="img"
            aria-label="HarnessML"
        >
            {rects}
        </svg>
    );
}

function SpeechBubble({ message }: { message: string | null }) {
    if (!message) return null;
    return (
        <div className={styles.speechBubble}>
            {message}
        </div>
    );
}

const I = { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };

// Overview
function IconDashboard() {
    return <svg {...I}><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>;
}
function IconDAG() {
    return <svg {...I}><circle cx="12" cy="4" r="2"/><circle cx="6" cy="20" r="2"/><circle cx="18" cy="20" r="2"/><line x1="12" y1="6" x2="6" y2="18"/><line x1="12" y1="6" x2="18" y2="18"/></svg>;
}
function IconConfig() {
    return <svg {...I}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>;
}

// Agent
function IconActivity() {
    return <svg {...I}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>;
}
function IconExperiments() {
    return <svg {...I}><path d="M9 3h6v7l4 9H5l4-9V3z"/><line x1="9" y1="3" x2="15" y2="3"/></svg>;
}
function IconNotebook() {
    return <svg {...I}><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/><line x1="8" y1="7" x2="16" y2="7"/><line x1="8" y1="11" x2="13" y2="11"/></svg>;
}

// Data
function IconSources() {
    return <svg {...I}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>;
}
function IconFeatures() {
    return <svg {...I}><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>;
}

// Models
function IconModels() {
    return <svg {...I}><rect x="4" y="4" width="16" height="16" rx="2"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="12" y1="4" x2="12" y2="20"/></svg>;
}
function IconEnsemble() {
    return <svg {...I}><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>;
}

// Analysis
function IconDiagnostics() {
    return <svg {...I}><path d="M21 21H4.6c-.56 0-.84 0-1.05-.11a1 1 0 01-.44-.44C3 20.24 3 19.96 3 19.4V3"/><path d="M7 14l4-4 4 4 6-6"/></svg>;
}
function IconPreferences() {
    return <svg {...I}><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>;
}
function IconPredictions() {
    return <svg {...I}><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>;
}

export function Layout() {
    const { project, '*': _rest } = useParams<{ project: string; '*': string }>();
    const projectName = project ?? '';
    const navigate = useNavigate();
    const location = useLocation();
    const currentPage = location.pathname.split('/').slice(2).join('/') || 'dashboard';
    const { events, connected } = useWebSocket();
    const { data: status } = useApi<ProjectStatus>('/api/project/status', undefined, projectName);
    const [projects, setProjects] = useState<ProjectInfo[]>([]);

    const robotStatus: RobotStatus = !connected
        ? 'offline'
        : events.some(e => e.status === 'running')
            ? 'working'
            : 'idle';

    const { addToast } = useToast();
    const [robotMessage, setRobotMessage] = useState<string | null>(null);
    const lastEventCount = useRef(events.length);
    const toastEventCount = useRef(events.length);

    useEffect(() => {
        if (events.length > lastEventCount.current) {
            const latest = events[events.length - 1];
            const msg = latest.status === 'running'
                ? `${latest.tool}.${latest.action}`
                : `${latest.tool}.${latest.action} done`;
            setRobotMessage(msg);
            const timer = setTimeout(() => setRobotMessage(null), 3000);
            lastEventCount.current = events.length;
            return () => clearTimeout(timer);
        }
        lastEventCount.current = events.length;
    }, [events.length]);

    // Toast on run completion
    useEffect(() => {
        if (events.length <= toastEventCount.current) {
            toastEventCount.current = events.length;
            return;
        }
        const newEvents = events.slice(toastEventCount.current);
        toastEventCount.current = events.length;
        for (const evt of newEvents) {
            if (evt.tool === 'pipeline' && evt.action === 'run_backtest' && (evt.status === 'success' || evt.status === 'error')) {
                if (evt.status === 'success') {
                    addToast('Run completed — metrics updated', 'success');
                } else {
                    addToast('Run failed — check logs for details', 'warning');
                }
            }
            if (evt.tool === 'experiments' && evt.action === 'quick_run' && (evt.status === 'success' || evt.status === 'error')) {
                if (evt.status === 'success') {
                    addToast('Experiment run completed', 'success');
                } else {
                    addToast('Experiment run failed', 'warning');
                }
            }
        }
    }, [events, addToast]);

    useEffect(() => {
        fetch('/api/projects')
            .then(r => r.json())
            .then(setProjects)
            .catch(() => {});
    }, []);

    const [showShortcuts, setShowShortcuts] = useState(false);

    const shortcutHandlers = useCallback(() => ({
        dashboard: () => navigate(`/${projectName}/dashboard`),
        activity: () => navigate(`/${projectName}/activity`),
        dag: () => navigate(`/${projectName}/dag`),
        experiments: () => navigate(`/${projectName}/experiments`),
        notebook: () => navigate(`/${projectName}/notebook`),
        diagnostics: () => navigate(`/${projectName}/diagnostics`),
        predictions: () => navigate(`/${projectName}/predictions`),
        refresh: () => window.location.reload(),
        help: () => setShowShortcuts(prev => !prev),
    }), [navigate, projectName]);

    useKeyboardShortcuts(shortcutHandlers());

    const sections = [
        {
            label: 'Overview',
            tabs: [
                { path: `/${projectName}/dashboard`, label: 'Dashboard', icon: <IconDashboard /> },
                { path: `/${projectName}/dag`, label: 'DAG', icon: <IconDAG /> },
            ],
        },
        {
            label: 'Agent',
            tabs: [
                { path: `/${projectName}/activity`, label: 'Activity', icon: <IconActivity />, badge: 'MCP' },
                { path: `/${projectName}/experiments`, label: 'Experiments', icon: <IconExperiments /> },
                { path: `/${projectName}/notebook`, label: 'Notebook', icon: <IconNotebook /> },
            ],
        },
        {
            label: 'Data',
            tabs: [
                { path: `/${projectName}/data`, label: 'Sources', icon: <IconSources /> },
                { path: `/${projectName}/features`, label: 'Features', icon: <IconFeatures /> },
            ],
        },
        {
            label: 'Models',
            tabs: [
                { path: `/${projectName}/models`, label: 'Base', icon: <IconModels /> },
                { path: `/${projectName}/ensemble`, label: 'Ensemble', icon: <IconEnsemble /> },
            ],
        },
        {
            label: 'Analysis',
            tabs: [
                { path: `/${projectName}/diagnostics`, label: 'Diagnostics', icon: <IconDiagnostics /> },
                { path: `/${projectName}/predictions`, label: 'Predictions', icon: <IconPredictions /> },
            ],
        },
        {
            label: 'System',
            tabs: [
                { path: `/${projectName}/config`, label: 'Config', icon: <IconConfig /> },
                { path: `/${projectName}/preferences`, label: 'Preferences', icon: <IconPreferences /> },
            ],
        },
    ];

    return (
        <ProjectContext.Provider value={projectName}>
            <div className={styles.layout}>
                <header className={styles.header}>
                    <div className={styles.headerLeft}>
                        <div className={styles.logoBlock}>
                            <div className={styles.logoRow}>
                                <Link to={`/${projectName}/dashboard`} className={styles.logoLink}>
                                    <AsciiLogo robotStatus={robotStatus} />
                                </Link>
                                <SpeechBubble message={robotMessage} />
                            </div>
                            <span className={styles.logoSubtitle}>An Agent-Computer Interface for Machine Learning</span>
                        </div>
                    </div>
                    <div className={styles.headerCenter} />
                    <div className={styles.headerRight}>
                        {status?.task && status.task !== 'unknown' && (
                            <div className={styles.headerStat}>
                                <span className={styles.headerStatLabel}>Task</span>
                                <span className={styles.headerStatValue}>{status.task}</span>
                            </div>
                        )}
                        {status?.target_column && (
                            <div className={styles.headerStat}>
                                <span className={styles.headerStatLabel}>Target</span>
                                <span className={styles.headerStatValue}>{status.target_column}</span>
                            </div>
                        )}
                        {status?.latest_metrics && Object.keys(status.latest_metrics).length > 0 && (
                            <>
                                <span className={styles.divider} />
                                {Object.entries(status.latest_metrics).map(([name, value]) => (
                                    <div key={name} className={styles.headerStat}>
                                        <span className={styles.headerStatLabel}>{name}</span>
                                        <span className={styles.headerStatValue}>
                                            {Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4)}
                                        </span>
                                    </div>
                                ))}
                            </>
                        )}
                        {(status?.task || status?.target_column) && <span className={styles.divider} />}
                        {projects.length > 0 ? (
                            <div className={styles.projectSelector}>
                                <span className={styles.projectSelectorLabel}>Project</span>
                                <select
                                    className={styles.projectSelect}
                                    value={projectName}
                                    onChange={(e) => navigate(`/${e.target.value}/${currentPage}`)}
                                >
                                    {projects.map(p => (
                                        <option key={p.name} value={p.name}>{p.name}</option>
                                    ))}
                                </select>
                            </div>
                        ) : projectName ? (
                            <div className={styles.projectSelector}>
                                <span className={styles.projectSelectorLabel}>Project</span>
                                <span className={styles.projectName}>{projectName}</span>
                            </div>
                        ) : null}
                    </div>
                </header>
                <div className={styles.body}>
                    <nav className={styles.nav}>
                        {sections.map(section => (
                            <div key={section.label} className={styles.navSection}>
                                <div className={styles.navSectionLabel}>{section.label}</div>
                                {section.tabs.map(t => (
                                    <NavLink key={t.path} to={t.path} end
                                        className={({ isActive }) =>
                                            isActive ? styles.tabActive : styles.tab
                                        }>
                                        {t.icon}
                                        {t.label}
                                        {'badge' in t && t.badge && (
                                            <span className={styles.navBadge}>{t.badge}</span>
                                        )}
                                    </NavLink>
                                ))}
                            </div>
                        ))}
                        <div className={styles.navSpacer} />
                        <div className={styles.navStatus}>
                            <span className={connected ? styles.statusLive : styles.statusOffline}>
                                <span className={styles.statusDot} />
                                {connected ? 'Connected' : 'Offline'}
                            </span>
                        </div>
                    </nav>
                    <main className={styles.content}>
                        <Outlet context={{ events, connected } satisfies LayoutContext} />
                    </main>
                </div>
                {showShortcuts && (
                    <div className={styles.shortcutsOverlay} onClick={() => setShowShortcuts(false)}>
                        <div className={styles.shortcutsPanel} onClick={e => e.stopPropagation()}>
                            <div className={styles.shortcutsPanelHeader}>
                                <span className={styles.shortcutsPanelTitle}>Keyboard Shortcuts</span>
                                <button className={styles.shortcutsPanelClose} onClick={() => setShowShortcuts(false)}>
                                    &times;
                                </button>
                            </div>
                            <div className={styles.shortcutsList}>
                                {SHORTCUT_DESCRIPTIONS.map(s => (
                                    <div key={s.key} className={styles.shortcutRow}>
                                        <kbd className={styles.shortcutKey}>{s.key}</kbd>
                                        <span className={styles.shortcutLabel}>{s.label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </ProjectContext.Provider>
    );
}
