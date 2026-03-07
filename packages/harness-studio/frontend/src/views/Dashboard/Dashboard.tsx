import { useMemo, useState, useEffect, useRef } from 'react';
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
    ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import type { Event } from '../../hooks/useWebSocket';
import styles from './Dashboard.module.css';

// -- Types --

interface ProjectStatus {
    project_name: string;
    task: string;
    model_types_tried: number;
    active_models: number;
    experiments_run: number;
    run_count: number;
    feature_count: number;
    latest_metrics: Record<string, number>;
}

interface Experiment {
    experiment_id: string;
    timestamp?: string;
    hypothesis?: string;
    verdict?: string;
    primary_delta?: number;
    primary_metric?: string;
    metrics?: Record<string, number>;
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

function MiniNode({ data, type }: NodeProps) {
    const typeClass = MINI_TYPE_CLASS[type ?? ''] ?? '';
    return (
        <div className={`${styles.miniNode} ${typeClass}`}>
            <Handle type="target" position={Position.Left} style={{ visibility: 'hidden' }} />
            {String(data.label ?? '')}
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

function miniLayout(apiNodes: ApiNode[], apiEdges: ApiEdge[]): { nodes: Node[]; edges: Edge[] } {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'LR', nodesep: 20, ranksep: 50 });

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
        style: { stroke: '#484f58', strokeWidth: 1.5 },
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

const VERDICT_COLORS: Record<string, string> = {
    keep: '#3fb950',
    improved: '#3fb950',
    partial: '#d29922',
    neutral: '#d29922',
    revert: '#f85149',
    regressed: '#f85149',
};

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
    switch (verdict) {
        case 'keep':
        case 'improved': return styles.verdictKeep;
        case 'partial':
        case 'neutral': return styles.verdictPartial;
        case 'revert':
        case 'regressed': return styles.verdictRevert;
        default: return '';
    }
}

function formatMetric(val: number): string {
    if (Math.abs(val) >= 100) return val.toFixed(1);
    return val.toFixed(4);
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

// ===== TOP ROW: Project Overview (left) + Performance (right) =====

function ProjectOverviewWidget({ status, experiments, latestRun }: {
    status: ProjectStatus | null;
    experiments: Experiment[];
    latestRun: RunSummary | null;
}) {
    // Best performance metrics
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

    const bestExperiment = useMemo(() => {
        if (!primaryMetric) return null;
        let best: Experiment | null = null;
        let bestVal: number | null = null;
        for (const exp of experiments) {
            const val = exp.metrics?.[primaryMetric];
            if (val == null) continue;
            if (bestVal === null
                || (isLowerBetter && val < bestVal)
                || (!isLowerBetter && val > bestVal)
            ) {
                bestVal = val;
                best = exp;
            }
        }
        return best;
    }, [experiments, primaryMetric, isLowerBetter]);

    const displayMetrics = bestExperiment?.metrics ?? latestRun?.metrics ?? null;

    // Verdict breakdown for bar chart
    const verdictData = useMemo(() => {
        const counts: Record<string, number> = {};
        for (const exp of experiments) {
            const v = exp.verdict ?? 'pending';
            const key = v === 'improved' ? 'keep' : v === 'regressed' ? 'revert' : v;
            counts[key] = (counts[key] ?? 0) + 1;
        }
        return Object.entries(counts).map(([verdict, count]) => ({
            verdict,
            count,
            color: VERDICT_COLORS[verdict] ?? 'var(--color-text-muted)',
        }));
    }, [experiments]);

    return (
        <div className={`${styles.widget} ${styles.overviewWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Project Overview</span>
                <Link to="/diagnostics" className={styles.widgetLink}>Diagnostics</Link>
            </div>
            <div className={styles.widgetBody}>
                <div className={styles.overviewGrid}>
                    <div className={styles.overviewStat}>
                        <span className={styles.overviewValue}>{status?.feature_count ?? 0}</span>
                        <span className={styles.overviewLabel}>Features</span>
                    </div>
                    <div className={styles.overviewStat}>
                        <span className={styles.overviewValue}>{status?.active_models ?? 0}</span>
                        <span className={styles.overviewLabel}>Models</span>
                    </div>
                    <div className={styles.overviewStat}>
                        <span className={styles.overviewValue}>{status?.model_types_tried ?? 0}</span>
                        <span className={styles.overviewLabel}>Types Tried</span>
                    </div>
                    <div className={styles.overviewStat}>
                        <span className={styles.overviewValue}>{status?.run_count ?? 0}</span>
                        <span className={styles.overviewLabel}>Runs</span>
                    </div>
                </div>
                {displayMetrics && (
                    <div className={styles.latestMetrics}>
                        {Object.entries(displayMetrics).map(([name, value]) => (
                            <div key={name} className={styles.latestMetricItem}>
                                <span className={styles.latestMetricValue}>{formatMetric(value)}</span>
                                <span className={styles.latestMetricName}>{name}</span>
                            </div>
                        ))}
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
            </div>
        </div>
    );
}

function ExperimentsWidget({ experiments, latestRun }: {
    experiments: Experiment[];
    latestRun: RunSummary | null;
}) {
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
                verdict: exp.verdict,
            }));
    }, [experiments, primaryMetric]);

    const allValues = trendData.map(d => d.value);
    const yMin = allValues.length > 0 ? Math.min(...allValues) : 0;
    const yMax = allValues.length > 0 ? Math.max(...allValues) : 1;
    const padding = (yMax - yMin) * 0.2 || 0.01;
    const bestValue = allValues.length > 0
        ? (isLowerBetter ? Math.min(...allValues) : Math.max(...allValues))
        : null;

    const recent = useMemo(() => {
        return [...experiments]
            .sort((a, b) => (b.timestamp ?? '').localeCompare(a.timestamp ?? ''))
            .slice(0, 8);
    }, [experiments]);

    return (
        <div className={`${styles.widget} ${styles.experimentsWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Experiments</span>
                <Link to="/experiments" className={styles.widgetLink}>View all</Link>
            </div>
            <div className={styles.widgetBody}>
                {trendData.length >= 2 && primaryMetric && (
                    <div className={styles.trendChart}>
                        <div className={styles.trendLabel}>{primaryMetric} across experiments</div>
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
                                        const color = payload.verdict
                                            ? (VERDICT_COLORS[payload.verdict] ?? 'var(--color-accent)')
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
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}
                {recent.length === 0 ? (
                    <div className={styles.emptyState}>No experiments yet</div>
                ) : (
                    <ul className={styles.miniExpList}>
                        {recent.map(exp => (
                            <li key={exp.experiment_id} className={styles.miniExpRow}>
                                <span className={styles.miniExpId}>{exp.experiment_id}</span>
                                <span className={styles.miniExpHypothesis}>
                                    {exp.hypothesis || '--'}
                                </span>
                                <span className={`${styles.miniExpVerdict} ${verdictClass(exp.verdict)}`}>
                                    {exp.verdict ?? '--'}
                                </span>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

// ===== BOTTOM WIDGETS =====

function ActivityWidget({ events }: { events: Event[] }) {
    const merged = useMemo(() => mergeEventsForDashboard(events), [events]);
    const recent = merged.slice(0, 12);

    return (
        <div className={`${styles.widget} ${styles.activityWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Activity</span>
                <Link to="/activity" className={styles.widgetLink}>View all</Link>
            </div>
            <div className={styles.widgetBody}>
                {recent.length === 0 ? (
                    <div className={styles.emptyState}>No events yet</div>
                ) : (
                    recent.map(event => (
                        <div
                            key={event.id}
                            className={`${styles.miniEventRow} ${event.status === 'running' ? styles.miniEventRunning : ''}`}
                        >
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
                            {event.status !== 'running' && (
                                <span className={styles.miniEventDuration}>{event.duration_ms}ms</span>
                            )}
                            {event.status === 'running' && (
                                <span className={styles.miniEventDuration} style={{ color: 'var(--color-accent)' }}>
                                    ...
                                </span>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

function DagWidget({ refreshKey }: { refreshKey?: number }) {
    const { data, loading } = useApi<DagResponse>('/api/project/dag', refreshKey);
    const bodyRef = useRef<HTMLDivElement>(null);
    const [sizeKey, setSizeKey] = useState(0);

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
        return miniLayout(data.nodes, data.edges);
    }, [data]);

    return (
        <div className={`${styles.widget} ${styles.dagWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Pipeline DAG</span>
                <Link to="/dag" className={styles.widgetLink}>Full view</Link>
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
                    >
                        <Background color="#30363d" gap={20} size={1} />
                    </ReactFlow>
                )}
            </div>
        </div>
    );
}

// ===== Main Dashboard =====

export function Dashboard() {
    const { events } = useLayoutContext();

    // Refresh keys — re-fetch API data whenever relevant tool calls complete
    const allKey = useRefreshKey(events);
    const dagKey = useRefreshKey(events, ['models', 'features', 'config', 'pipeline']);
    const expKey = useRefreshKey(events, ['experiments', 'pipeline']);

    const { data: status } = useApi<ProjectStatus>('/api/project/status', allKey);
    const { data: runs } = useApi<RunSummary[]>('/api/runs', expKey);
    const { data: experiments } = useApi<Experiment[]>('/api/experiments', expKey);

    const latestRun = useMemo(() => {
        if (!runs || runs.length === 0) return null;
        return runs[0];
    }, [runs]);

    const exps = experiments ?? [];

    return (
        <div className={styles.dashboard}>
            <ProjectOverviewWidget status={status} experiments={exps} latestRun={latestRun} />
            <ExperimentsWidget experiments={exps} latestRun={latestRun} />
            <ActivityWidget events={events} />
            <DagWidget refreshKey={dagKey} />
        </div>
    );
}
