import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import {
    ReactFlow,
    Background,
    Handle,
    Position,
    type Node,
    type Edge,
    type NodeProps,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine,
    ResponsiveContainer, CartesianGrid, ErrorBar,
} from 'recharts';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useProject } from '../../hooks/useProject';
import type { Event } from '../../hooks/useWebSocket';
import { useTheme } from '../../hooks/useTheme';
import type { ThemeColors } from '../../styles/colors';
import { MetricLabel } from '../../components/Tooltip/Tooltip';
import type { NotebookEntry } from '../../types/api';
import { TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import styles from './Dashboard.module.css';

// -- Types --

interface Experiment {
    experiment_id: string;
    timestamp?: string;
    hypothesis?: string;
    verdict?: string;
    primary_delta?: number;
    primary_metric?: string;
    metrics?: Record<string, number>;
    metric_std?: Record<string, number>;
    baseline_metrics?: Record<string, number>;
}

interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    meta_coefficients?: Record<string, number>;
}

interface ApiNode {
    id: string;
    type: string;
    label: string;
    data: Record<string, unknown>;
}

interface ApiEdge {
    source: string;
    target: string;
}

interface DagResponse {
    nodes: ApiNode[];
    edges: ApiEdge[];
}

// -- Mini DAG node (label-only) --

const MINI_TYPE_CLASS: Record<string, string> = {
    source: styles.miniNodeSource,
    features: styles.miniNodeFeatures,
    model: styles.miniNodeModel,
    ensemble: styles.miniNodeEnsemble,
    calibration: styles.miniNodeCalibration,
    output: styles.miniNodeOutput,
    view: styles.miniNodeView,
};

function miniNodeDetails(type: string, data: Record<string, unknown>): string[] {
    const lines: string[] = [];
    switch (type) {
        case 'source':
            if (data.adapter) lines.push(`adapter: ${data.adapter}`);
            if (data.rows) lines.push(`${data.rows} rows`);
            if (Array.isArray(data.columns) && data.columns.length) lines.push(`${data.columns.length} columns`);
            if (data.path) lines.push(String(data.path));
            break;
        case 'features':
            if (data.count) lines.push(`${data.count} features`);
            if (data.task) lines.push(`task: ${data.task}`);
            if (data.target_column) lines.push(`target: ${data.target_column}`);
            if (Array.isArray(data.engineered_features) && data.engineered_features.length)
                lines.push(`${data.engineered_features.length} engineered`);
            break;
        case 'model':
            if (data.model_type) lines.push(String(data.model_type));
            if (data.mode) lines.push(`mode: ${data.mode}`);
            if (data.feature_count) lines.push(`${data.feature_count} features`);
            if (!data.active) lines.push('INACTIVE');
            break;
        case 'ensemble':
            if (data.method) lines.push(`method: ${data.method}`);
            if (data.model_count) lines.push(`${data.model_count} models`);
            if (data.cv_strategy) lines.push(`cv: ${data.cv_strategy}`);
            break;
        case 'calibration':
            if (data.type) lines.push(`type: ${data.type}`);
            break;
        case 'view':
            if (data.step_count) lines.push(`${data.step_count} steps`);
            if (Array.isArray(data.steps) && data.steps.length)
                lines.push(data.steps.join(' → '));
            break;
        case 'output':
            if (data.target) lines.push(`target: ${data.target}`);
            break;
    }
    return lines;
}

function MiniNode({ data, type }: NodeProps) {
    const typeClass = MINI_TYPE_CLASS[type ?? ''] ?? '';
    const isExperiment = !!(data as Record<string, unknown>)._experiment;
    const experimentClass = isExperiment ? styles.miniNodeExperiment : '';
    return (
        <div className={`${styles.miniNode} ${typeClass} ${experimentClass}`}>
            <Handle type="target" position={Position.Left} style={{ visibility: 'hidden' }} />
            {String(data.label ?? '')}
            {isExperiment && <span className={styles.miniExpBadge}>EXP</span>}
            <Handle type="source" position={Position.Right} style={{ visibility: 'hidden' }} />
        </div>
    );
}

const miniNodeTypes = {
    source: MiniNode,
    features: MiniNode,
    model: MiniNode,
    ensemble: MiniNode,
    calibration: MiniNode,
    output: MiniNode,
    view: MiniNode,
};

const MINI_NODE_WIDTH = 120;

function miniLayout(apiNodes: ApiNode[], apiEdges: ApiEdge[], colors: ThemeColors): { nodes: Node[]; edges: Edge[] } {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'LR', nodesep: 5, ranksep: 30, edgesep: 5 });

    const NODE_HEIGHT = 28;
    apiNodes.forEach(n => {
        g.setNode(n.id, { width: MINI_NODE_WIDTH, height: NODE_HEIGHT });
    });
    apiEdges.forEach(e => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const nodes: Node[] = apiNodes.map(n => {
        const pos = g.node(n.id);
        return {
            id: n.id,
            type: n.type,
            position: { x: pos.x - MINI_NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
            data: { label: n.label, ...n.data },
        };
    });

    const edges: Edge[] = apiEdges.map((e, i) => ({
        id: `e${i}`,
        source: e.source,
        target: e.target,
        type: 'smoothstep',
        style: { stroke: colors.border, strokeWidth: 1.5 },
        animated: false,
    }));

    return { nodes, edges };
}

// -- Tool badge colors --

const TOOL_COLOR_MAP: Record<string, string> = {
    models: 'var(--color-tool-models)',
    pipeline: 'var(--color-tool-pipeline)',
    data: 'var(--color-tool-data)',
    features: 'var(--color-tool-features)',
    experiments: 'var(--color-tool-experiments)',
    config: 'var(--color-tool-config)',
};

// -- Helpers --

const LOWER_IS_BETTER = new Set(['brier', 'ece', 'log_loss', 'mae', 'mse', 'rmse']);

function getVerdictColors(c: ThemeColors): Record<string, string> {
    return {
        positive: c.success,
        neutral: c.warning,
        negative: c.error,
        pending: c.textMuted,
    };
}

/** Normalize any verdict string into positive / neutral / negative / pending. */
function normalizeVerdict(v: string | undefined): string {
    if (!v) return 'pending';
    const lower = v.toLowerCase();
    if (['keep', 'improved', 'positive'].includes(lower)) return 'positive';
    if (['revert', 'regressed', 'negative', 'failed'].includes(lower)) return 'negative';
    if (['partial', 'neutral', 'baseline'].includes(lower)) return 'neutral';
    return 'pending';
}

function formatRelativeTime(timestamp: string): string {
    const diffMs = Date.now() - new Date(timestamp).getTime();
    if (diffMs < 0 || diffMs < 10000) return 'now';
    const seconds = Math.floor(diffMs / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h`;
    return `${Math.floor(hours / 24)}d`;
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


function mergeEventsForDashboard(events: Event[]): Event[] {
    const chronological = [...events].sort((a, b) => {
        const ta = new Date(a.timestamp).getTime();
        const tb = new Date(b.timestamp).getTime();
        if (ta !== tb) return ta - tb;
        return a.id - b.id;
    });

    const openGroups = new Map<string, Event>();
    const merged: Event[] = [];

    for (const event of chronological) {
        const key = `${event.tool}:${event.action}`;
        if (event.status === 'running') {
            openGroups.set(key, { ...event });
            continue;
        }
        if (event.status === 'progress') {
            const group = openGroups.get(key);
            if (group) {
                group.result = event.result;
                group.params = event.params;
            }
            continue;
        }
        if (event.status === 'success' || event.status === 'error') {
            if (openGroups.has(key)) {
                openGroups.delete(key);
            }
        }
        merged.push(event);
    }
    for (const group of openGroups.values()) {
        merged.push(group);
    }

    merged.sort((a, b) => {
        const ta = new Date(a.timestamp).getTime();
        const tb = new Date(b.timestamp).getTime();
        if (tb !== ta) return tb - ta;
        return b.id - a.id;
    });

    return merged;
}


function ExperimentsWidget({ experiments, latestRun }: {
    experiments: Experiment[];
    latestRun: RunSummary | null;
}) {
    const { colors } = useTheme();
    const VERDICT_COLORS = useMemo(() => getVerdictColors(colors), [colors]);

    // Primary metric + trend chart (moved from old PerformanceWidget)
    const primaryMetric = useMemo(() => {
        for (const exp of experiments) {
            if (exp.primary_metric) return exp.primary_metric;
        }
        if (latestRun) {
            const keys = Object.keys(latestRun.metrics);
            if (keys.includes('brier')) return 'brier';
            if (keys.includes('accuracy')) return 'accuracy';
            return keys[0] ?? null;
        }
        return null;
    }, [experiments, latestRun]);

    const isLowerBetter = primaryMetric ? LOWER_IS_BETTER.has(primaryMetric) : false;

    const trendData = useMemo(() => {
        if (!primaryMetric) return [];
        const chrono = [...experiments]
            .sort((a, b) => (a.timestamp ?? '').localeCompare(b.timestamp ?? ''));
        return chrono
            .filter(exp => exp.metrics?.[primaryMetric] != null)
            .map(exp => ({
                id: exp.experiment_id.replace('exp-', ''),
                value: exp.metrics![primaryMetric],
                errorY: exp.metric_std?.[primaryMetric] ?? 0,
                verdict: exp.verdict,
            }));
    }, [experiments, primaryMetric]);

    const hasErrorBars = trendData.some(d => d.errorY > 0);
    const baseValues = trendData.map(d => d.value);
    const bestValue = baseValues.length > 0
        ? (isLowerBetter ? Math.min(...baseValues) : Math.max(...baseValues))
        : null;
    const allValues = [...baseValues];
    if (hasErrorBars) {
        for (const d of trendData) {
            allValues.push(d.value + d.errorY);
            allValues.push(d.value - d.errorY);
        }
    }
    const yMin = allValues.length > 0 ? Math.min(...allValues) : 0;
    const yMax = allValues.length > 0 ? Math.max(...allValues) : 1;
    const padding = (yMax - yMin) * 0.2 || 0.01;

    const VERDICT_ORDER = ['positive', 'neutral', 'pending', 'negative'];
    const verdictData = useMemo(() => {
        const counts: Record<string, number> = {};
        for (const exp of experiments) {
            const key = normalizeVerdict(exp.verdict);
            counts[key] = (counts[key] ?? 0) + 1;
        }
        return VERDICT_ORDER
            .filter(v => counts[v])
            .map(verdict => ({
                verdict,
                count: counts[verdict],
                color: VERDICT_COLORS[verdict] ?? 'var(--color-text-muted)',
            }));
    }, [experiments, VERDICT_COLORS]);

    const recent = useMemo(() => {
        return [...experiments]
            .sort((a, b) => (b.timestamp ?? '').localeCompare(a.timestamp ?? ''))
            .slice(0, 8);
    }, [experiments]);

    return (
        <div className={`${styles.widget} ${styles.experimentsWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Experiments</span>
                <Link to="../experiments" className={styles.widgetLink}>View all</Link>
            </div>
            <div className={styles.widgetBody}>
                {trendData.length >= 2 && primaryMetric && (
                    <div className={styles.trendChart}>
                        <div className={styles.trendLabel}><MetricLabel name={primaryMetric} /> across experiments</div>
                        <ResponsiveContainer width="100%" height={120}>
                            <LineChart data={trendData} margin={{ top: 4, right: 12, bottom: 4, left: 12 }}>
                                <CartesianGrid
                                    stroke="var(--color-border-subtle)"
                                    strokeDasharray="3 3"
                                    vertical={false}
                                />
                                <XAxis
                                    dataKey="id"
                                    tick={{ fill: 'var(--color-text-muted)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <YAxis
                                    domain={[yMin - padding, yMax + padding]}
                                    tick={{ fill: 'var(--color-text-muted)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                                    axisLine={false}
                                    tickLine={false}
                                    tickFormatter={(v: number) => v.toFixed(3)}
                                    width={42}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: 'var(--color-bg-tertiary)',
                                        border: '1px solid var(--color-border)',
                                        borderRadius: 'var(--radius-sm)',
                                        color: 'var(--color-text-primary)',
                                        fontFamily: 'var(--font-mono)',
                                        fontSize: 11,
                                    }}
                                    formatter={(value) => [Number(value).toFixed(4), primaryMetric]}
                                />
                                {bestValue != null && (
                                    <ReferenceLine
                                        y={bestValue}
                                        stroke="var(--color-success)"
                                        strokeDasharray="4 2"
                                        strokeWidth={1}
                                    />
                                )}
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="var(--color-accent)"
                                    strokeWidth={2}
                                    dot={(props: Record<string, unknown>) => {
                                        const { cx, cy, payload } = props as {
                                            cx: number; cy: number;
                                            payload: { verdict?: string };
                                        };
                                        const normalized = normalizeVerdict(payload.verdict);
                                        const color = normalized !== 'pending'
                                            ? (VERDICT_COLORS[normalized] ?? 'var(--color-accent)')
                                            : 'var(--color-accent)';
                                        return (
                                            <circle
                                                key={`dot-${cx}-${cy}`}
                                                cx={cx} cy={cy} r={3}
                                                fill={color} stroke={color} strokeWidth={1}
                                            />
                                        );
                                    }}
                                    activeDot={{ r: 5 }}
                                >
                                    {hasErrorBars && (
                                        <ErrorBar
                                            dataKey="errorY"
                                            direction="y"
                                            width={3}
                                            stroke="var(--color-text-muted)"
                                            strokeWidth={1}
                                        />
                                    )}
                                </Line>
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}
                {verdictData.length > 0 && (
                    <div className={styles.verdictSection}>
                        <div className={styles.verdictBarLabel}>
                            {experiments.length} experiments
                        </div>
                        <div className={styles.verdictBar}>
                            {verdictData.map(d => (
                                <div
                                    key={d.verdict}
                                    className={styles.verdictSegment}
                                    style={{
                                        flex: d.count,
                                        backgroundColor: d.color,
                                    }}
                                    title={`${d.verdict}: ${d.count}`}
                                />
                            ))}
                        </div>
                        <div className={styles.verdictLegend}>
                            {verdictData.map(d => (
                                <span key={d.verdict} className={styles.verdictLegendItem}>
                                    <span className={styles.verdictDot} style={{ background: d.color }} />
                                    {d.verdict} ({d.count})
                                </span>
                            ))}
                        </div>
                    </div>
                )}
                {recent.length === 0 ? (
                    <div className={styles.emptyState}>No experiments yet</div>
                ) : (
                    <ul className={styles.miniExpList}>
                        {recent.map((exp, i) => (
                            <li key={`${exp.experiment_id}-${i}`} className={styles.miniExpRow}>
                                <Link
                                    to="../experiments"
                                    state={{ expandedId: exp.experiment_id }}
                                    className={styles.miniExpLink}
                                >
                                    <span className={styles.miniExpId}>{exp.experiment_id}</span>
                                    <span className={styles.miniExpHypothesis}>
                                        {exp.hypothesis || '--'}
                                    </span>
                                    <span className={`${styles.miniExpVerdict} ${verdictClass(exp.verdict)}`}>
                                        {exp.verdict ?? '--'}
                                    </span>
                                </Link>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

// ===== BOTTOM WIDGETS =====

function ActivityStrip({ events }: { events: Event[] }) {
    const merged = useMemo(() => mergeEventsForDashboard(events), [events]);
    const latest = merged[0] ?? null;
    const previous = merged[1] ?? null;

    function EventRow({ event, faded }: { event: Event; faded?: boolean }) {
        return (
            <div className={`${styles.stripRow} ${faded ? styles.stripRowFaded : ''} ${event.status === 'running' ? styles.stripRowRunning : ''}`}>
                <span className={styles.miniEventTime}>
                    {formatRelativeTime(event.timestamp)}
                </span>
                <span
                    className={styles.miniEventTool}
                    style={{ backgroundColor: TOOL_COLOR_MAP[event.tool] ?? 'var(--color-text-muted)' }}
                >
                    {event.tool}
                </span>
                <span className={styles.miniEventAction}>{event.action}</span>
                {event.caller && (
                    <span className={styles.miniEventCaller}>{event.caller}</span>
                )}
                {event.status !== 'running' && (
                    <span className={styles.miniEventDuration}>{event.duration_ms}ms</span>
                )}
                {event.status === 'running' && (
                    <span className={styles.miniEventDuration} style={{ color: 'var(--color-accent)' }}>...</span>
                )}
            </div>
        );
    }

    return (
        <div className={styles.activityStrip}>
            <Link to="../activity" className={styles.stripLabel}>
                <span className={styles.mcpBadge}>MCP</span>
            </Link>
            <div className={styles.stripEvents}>
                {latest ? (
                    <>
                        <EventRow event={latest} />
                        {previous && <EventRow event={previous} faded />}
                    </>
                ) : (
                    <span className={styles.stripIdle}>Listening for tool calls...</span>
                )}
            </div>
        </div>
    );
}

function DagWidget({ refreshKey }: { refreshKey?: number }) {
    const { colors } = useTheme();
    const project = useProject();
    const { data, loading } = useApi<DagResponse>('/api/project/dag', refreshKey, project);
    const bodyRef = useRef<HTMLDivElement>(null);
    const [sizeKey, setSizeKey] = useState(0);
    const [tooltip, setTooltip] = useState<{ x: number; y: number; lines: string[] } | null>(null);

    // Remount ReactFlow when container resizes so fitView recalculates
    useEffect(() => {
        const el = bodyRef.current;
        if (!el) return;
        let timer = 0;
        const observer = new ResizeObserver(() => {
            clearTimeout(timer);
            timer = window.setTimeout(() => setSizeKey(k => k + 1), 150);
        });
        observer.observe(el);
        return () => { clearTimeout(timer); observer.disconnect(); };
    }, []);

    const { nodes, edges } = useMemo(() => {
        if (!data) return { nodes: [], edges: [] };
        return miniLayout(data.nodes, data.edges, colors);
    }, [data, colors]);

    // Re-fit when DAG data changes (new nodes/edges from backend)
    useEffect(() => {
        if (data) {
            setSizeKey(k => k + 1);
        }
    }, [data]);

    const handleNodeMouseEnter = useCallback((_: React.MouseEvent, node: Node) => {
        const lines = miniNodeDetails(node.type ?? '', node.data as Record<string, unknown>);
        if (lines.length === 0) return;
        const el = document.querySelector(`[data-id="${node.id}"]`);
        if (!el) return;
        const rect = el.getBoundingClientRect();
        setTooltip({ x: rect.left + rect.width / 2, y: rect.top, lines });
    }, []);

    const handleNodeMouseLeave = useCallback(() => {
        setTooltip(null);
    }, []);

    return (
        <div className={`${styles.widget} ${styles.dagWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Pipeline DAG</span>
                <Link to="../dag" className={styles.widgetLink}>Full view</Link>
            </div>
            <div className={styles.dagBody} ref={bodyRef}>
                {loading ? (
                    <div className={styles.emptyState}>Loading DAG...</div>
                ) : nodes.length === 0 ? (
                    <div className={styles.emptyState}>No pipeline configured</div>
                ) : (
                    <ReactFlow
                        key={sizeKey}
                        nodes={nodes}
                        edges={edges}
                        nodeTypes={miniNodeTypes}
                        fitView
                        fitViewOptions={{ padding: 0.05 }}
                        minZoom={0.1}
                        proOptions={{ hideAttribution: true }}
                        nodesDraggable={false}
                        nodesConnectable={false}
                        elementsSelectable={false}
                        panOnDrag={false}
                        zoomOnScroll={false}
                        zoomOnPinch={false}
                        zoomOnDoubleClick={false}
                        preventScrolling={false}
                        onNodeMouseEnter={handleNodeMouseEnter}
                        onNodeMouseLeave={handleNodeMouseLeave}
                    >
                        <Background color={colors.border} gap={20} size={1} />
                    </ReactFlow>
                )}
            </div>
            {tooltip && createPortal(
                <div
                    className={styles.miniNodeTooltip}
                    style={{ left: tooltip.x, top: tooltip.y }}
                >
                    {tooltip.lines.map((line, i) => <div key={i}>{line}</div>)}
                </div>,
                document.body,
            )}
        </div>
    );
}

// ===== Narrative Widget =====

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

function NarrativeWidget({ entries }: { entries: NotebookEntry[] }) {
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);
    const latestTheory = nonStruck.find(e => e.type === 'theory') ?? null;
    const latestPlan = nonStruck.find(e => e.type === 'plan') ?? null;

    return (
        <>
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
        </>
    );
}

// ===== Main Dashboard =====

export function Dashboard() {
    const { events } = useLayoutContext();
    const project = useProject();

    // Refresh keys — re-fetch API data whenever relevant tool calls complete
    const dagKey = useRefreshKey(events, ['models', 'features', 'config', 'pipeline']);
    const expKey = useRefreshKey(events, ['experiments', 'pipeline']);

    const nbKey = useRefreshKey(events, ['notebook']);

    const { data: runs } = useApi<RunSummary[]>('/api/runs', expKey, project);
    const { data: experiments } = useApi<Experiment[]>('/api/experiments', expKey, project);
    const { data: notebookEntries } = useApi<NotebookEntry[]>('/api/notebook', nbKey, project);
    const nbEntries = notebookEntries ?? [];

    const latestRun = useMemo(() => {
        if (!runs || runs.length === 0) return null;
        return runs[0];
    }, [runs]);

    const exps = experiments ?? [];

    return (
        <div className={styles.dashboard}>
            <ActivityStrip events={events} />
            <NarrativeWidget entries={nbEntries} />
            <ExperimentsWidget experiments={exps} latestRun={latestRun} />
            <DagWidget refreshKey={dagKey} />
        </div>
    );
}
