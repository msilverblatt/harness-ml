import { useMemo, useState, useCallback, useEffect, useRef, type ReactNode } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    type Node,
    type Edge,
    type NodeMouseHandler,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useWebSocket } from '../../hooks/useWebSocket';
import { SourceNode } from './nodes/SourceNode';
import { FeatureNode } from './nodes/FeatureNode';
import { ModelNode } from './nodes/ModelNode';
import { EnsembleNode } from './nodes/EnsembleNode';
import { CalibrationNode } from './nodes/CalibrationNode';
import { OutputNode } from './nodes/OutputNode';
import { ViewNode } from './nodes/ViewNode';
import styles from './DAG.module.css';

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

const nodeTypes = {
    source: SourceNode,
    features: FeatureNode,
    model: ModelNode,
    ensemble: EnsembleNode,
    calibration: CalibrationNode,
    output: OutputNode,
    view: ViewNode,
};

const NODE_WIDTH = 260;

function estimateNodeHeight(node: ApiNode): number {
    const d = node.data;
    let h = 50;

    switch (node.type) {
        case 'source': {
            if (d.path) h += 18;
            if ((d.rows as number) > 0) h += 18;
            const cols = (d.columns as string[]) ?? [];
            if (cols.length > 0) h += 18 + 10 + Math.ceil(Math.min(cols.length, 8) / 3) * 26 + (cols.length > 8 ? 22 : 0);
            return h;
        }
        case 'features': {
            if (d.target_column) h += 18;
            const raw = (d.raw_features as string[]) ?? [];
            const eng = (d.engineered_features as { name: string; formula: string }[]) ?? [];
            if (raw.length > 0) h += 10 + 18 + Math.ceil(Math.min(raw.length, 8) / 3) * 26 + (raw.length > 8 ? 22 : 0);
            if (eng.length > 0) h += 10 + 18 + Math.min(eng.length, 6) * 20 + (eng.length > 6 ? 18 : 0);
            return h;
        }
        case 'model': {
            h += 18;
            const params = d.params as Record<string, unknown> ?? {};
            const keyParams = ['max_depth', 'learning_rate', 'n_estimators', 'num_leaves', 'C', 'max_iter', 'dropout', 'hidden_layers'];
            const visibleParams = keyParams.filter(k => k in params).length;
            if (visibleParams > 0) h += 10 + Math.min(visibleParams, 5) * 20;
            const feats = (d.features as string[]) ?? [];
            if (feats.length > 0) h += 10 + Math.ceil(Math.min(feats.length, 6) / 3) * 26 + (feats.length > 6 ? 22 : 0);
            return h;
        }
        case 'ensemble': {
            if (d.cv_strategy) h += 10 + 20 + (d.fold_column ? 20 : 0);
            const metrics = (d.metrics as string[]) ?? [];
            if (metrics.length > 0) h += 10 + 18 + Math.ceil(metrics.length / 3) * 26;
            return h;
        }
        case 'calibration':
            return 80;
        case 'output':
            return d.target ? 70 : 50;
        case 'view': {
            const steps = (d.steps as string[]) ?? [];
            if (steps.length > 0) h += 10 + Math.ceil(steps.length / 3) * 26;
            return h;
        }
        default:
            return 80;
    }
}

function layoutWithDagre(apiNodes: ApiNode[], apiEdges: ApiEdge[]): { nodes: Node[]; edges: Edge[] } {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'LR', nodesep: 80, ranksep: 160 });

    apiNodes.forEach(n => {
        const h = estimateNodeHeight(n);
        g.setNode(n.id, { width: NODE_WIDTH, height: h });
    });
    apiEdges.forEach(e => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const nodes: Node[] = apiNodes.map(n => {
        const pos = g.node(n.id);
        const h = estimateNodeHeight(n);
        return {
            id: n.id,
            type: n.type,
            position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - h / 2 },
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

const MINIMAP_NODE_COLOR: Record<string, string> = {
    source: '#d2a8ff',
    features: '#79c0ff',
    model: '#58a6ff',
    ensemble: '#3fb950',
    calibration: '#d29922',
    output: '#f0883e',
    view: '#58a6ff',
};

function miniMapNodeColor(node: Node): string {
    return MINIMAP_NODE_COLOR[node.type ?? ''] ?? '#30363d';
}

function DetailPanel({
    node,
    onClose,
}: {
    node: Node;
    onClose: () => void;
}) {
    const d = node.data as Record<string, string | number | boolean | string[] | Record<string, unknown> | { name: string; formula: string }[] | undefined>;
    const nodeType = node.type ?? 'unknown';

    function renderModelDetail() {
        const params = (d.params as Record<string, unknown>) ?? {};
        const features = (d.features as string[]) ?? [];
        const isActive = d.active !== false;
        return (
            <>
                <div className={styles.detailSection}>
                    <div className={styles.detailSectionLabel}>Configuration</div>
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Type</span>
                        <span className={styles.detailValue}>{String(d.model_type ?? 'unknown')}</span>
                    </div>
                    {d.mode && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>Mode</span>
                            <span className={styles.detailValue}>{String(d.mode)}</span>
                        </div>
                    )}
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Status</span>
                        <span className={`${styles.statusBadge} ${isActive ? styles.statusActive : styles.statusInactive}`}>
                            {isActive ? 'Active' : 'Inactive'}
                        </span>
                    </div>
                </div>
                {Object.keys(params).length > 0 && (
                    <div className={styles.detailSection}>
                        <div className={styles.detailSectionLabel}>Parameters</div>
                        {Object.entries(params).map(([k, v]) => (
                            <div key={k} className={styles.detailRow}>
                                <span className={styles.detailKey}>{k}</span>
                                <span className={styles.detailValue}>{String(v)}</span>
                            </div>
                        ))}
                    </div>
                )}
                {features.length > 0 && (
                    <div className={styles.detailSection}>
                        <div className={styles.detailSectionLabel}>Features ({features.length})</div>
                        <div className={styles.detailTagList}>
                            {features.map(f => (
                                <span key={f} className={styles.detailTag}>{f}</span>
                            ))}
                        </div>
                    </div>
                )}
            </>
        );
    }

    function renderEnsembleDetail() {
        const metrics = (d.metrics as string[]) ?? [];
        return (
            <>
                <div className={styles.detailSection}>
                    <div className={styles.detailSectionLabel}>Configuration</div>
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Method</span>
                        <span className={styles.detailValue}>{String(d.method ?? 'stacking')}</span>
                    </div>
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Models</span>
                        <span className={styles.detailValue}>{String(d.model_count ?? 0)}</span>
                    </div>
                    {d.cv_strategy && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>CV Strategy</span>
                            <span className={styles.detailValue}>{String(d.cv_strategy)}</span>
                        </div>
                    )}
                    {d.fold_column && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>Fold Column</span>
                            <span className={styles.detailValue}>{String(d.fold_column)}</span>
                        </div>
                    )}
                </div>
                {metrics.length > 0 && (
                    <div className={styles.detailSection}>
                        <div className={styles.detailSectionLabel}>Metrics ({metrics.length})</div>
                        <div className={styles.detailTagList}>
                            {metrics.map(m => (
                                <span key={m} className={styles.detailTag}>{m}</span>
                            ))}
                        </div>
                    </div>
                )}
            </>
        );
    }

    function renderFeatureStoreDetail() {
        const raw = (d.raw_features as string[]) ?? [];
        const eng = (d.engineered_features as { name: string; formula: string }[]) ?? [];
        return (
            <>
                <div className={styles.detailSection}>
                    <div className={styles.detailSectionLabel}>Overview</div>
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Total</span>
                        <span className={styles.detailValue}>{String(d.count ?? 0)}</span>
                    </div>
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Raw</span>
                        <span className={styles.detailValue}>{raw.length}</span>
                    </div>
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Engineered</span>
                        <span className={styles.detailValue}>{eng.length}</span>
                    </div>
                    {d.target_column && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>Target</span>
                            <span className={styles.detailValue}>{String(d.target_column)}</span>
                        </div>
                    )}
                    {d.task && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>Task</span>
                            <span className={styles.detailValue}>{String(d.task)}</span>
                        </div>
                    )}
                </div>
                {raw.length > 0 && (
                    <div className={styles.detailSection}>
                        <div className={styles.detailSectionLabel}>Raw Features</div>
                        <div className={styles.detailTagList}>
                            {raw.map(f => (
                                <span key={f} className={styles.detailTag}>{f}</span>
                            ))}
                        </div>
                    </div>
                )}
                {eng.length > 0 && (
                    <div className={styles.detailSection}>
                        <div className={styles.detailSectionLabel}>Engineered Features</div>
                        {eng.map(f => (
                            <div key={f.name} className={styles.detailRow}>
                                <span className={styles.detailKey}>{f.name}</span>
                                <span className={styles.detailValue}>{f.formula}</span>
                            </div>
                        ))}
                    </div>
                )}
            </>
        );
    }

    function renderSourceDetail() {
        const columns = (d.columns as string[]) ?? [];
        return (
            <>
                <div className={styles.detailSection}>
                    <div className={styles.detailSectionLabel}>Source Info</div>
                    {d.path && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>Path</span>
                            <span className={styles.detailValue}>{String(d.path)}</span>
                        </div>
                    )}
                    {d.adapter && (
                        <div className={styles.detailRow}>
                            <span className={styles.detailKey}>Format</span>
                            <span className={styles.detailValue}>{String(d.adapter)}</span>
                        </div>
                    )}
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Rows</span>
                        <span className={styles.detailValue}>{(d.rows as number)?.toLocaleString?.() ?? '0'}</span>
                    </div>
                </div>
                {columns.length > 0 && (
                    <div className={styles.detailSection}>
                        <div className={styles.detailSectionLabel}>Columns ({columns.length})</div>
                        <div className={styles.detailTagList}>
                            {columns.map(c => (
                                <span key={c} className={styles.detailTag}>{c}</span>
                            ))}
                        </div>
                    </div>
                )}
            </>
        );
    }

    function renderCalibrationDetail() {
        return (
            <div className={styles.detailSection}>
                <div className={styles.detailSectionLabel}>Configuration</div>
                <div className={styles.detailRow}>
                    <span className={styles.detailKey}>Type</span>
                    <span className={styles.detailValue}>{String(d.type ?? 'unknown')}</span>
                </div>
            </div>
        );
    }

    function renderOutputDetail() {
        return (
            <div className={styles.detailSection}>
                <div className={styles.detailSectionLabel}>Output</div>
                {d.target && (
                    <div className={styles.detailRow}>
                        <span className={styles.detailKey}>Target</span>
                        <span className={styles.detailValue}>{String(d.target)}</span>
                    </div>
                )}
            </div>
        );
    }

    function renderViewDetail() {
        const steps = (d.steps as string[]) ?? [];
        return (
            <div className={styles.detailSection}>
                <div className={styles.detailSectionLabel}>Transform Steps ({steps.length})</div>
                <div className={styles.detailTagList}>
                    {steps.map((s, i) => (
                        <span key={i} className={styles.detailTag}>{s}</span>
                    ))}
                </div>
            </div>
        );
    }

    const renderers: Record<string, () => ReactNode> = {
        model: renderModelDetail,
        ensemble: renderEnsembleDetail,
        features: renderFeatureStoreDetail,
        source: renderSourceDetail,
        calibration: renderCalibrationDetail,
        output: renderOutputDetail,
        view: renderViewDetail,
    };

    const renderContent = renderers[nodeType];

    return (
        <div className={styles.detailPanel}>
            <div className={styles.detailPanelHeader}>
                <div>
                    <div className={styles.detailPanelTitle}>{String(d.label ?? node.id)}</div>
                    <div className={styles.detailNodeType}>{nodeType}</div>
                </div>
                <button className={styles.detailPanelClose} onClick={onClose}>X</button>
            </div>
            <hr className={styles.detailDivider} />
            {renderContent ? renderContent() : null}
        </div>
    );
}

export function DAG() {
    const { events } = useWebSocket();
    const dagRefreshKey = useRefreshKey(events, ['models', 'features', 'config', 'pipeline']);
    const { data, loading, error } = useApi<DagResponse>('/api/project/dag', dagRefreshKey);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [runningNodeIds, setRunningNodeIds] = useState<Set<string>>(new Set());
    const processedEventCountRef = useRef(0);
    const canvasRef = useRef<HTMLDivElement>(null);

    // Force ReactFlow to re-mount and fitView on container resize
    const [resizeKey, setResizeKey] = useState(0);
    useEffect(() => {
        const el = canvasRef.current;
        if (!el) return;
        let timer = 0;
        const observer = new ResizeObserver(() => {
            clearTimeout(timer);
            timer = window.setTimeout(() => {
                setResizeKey(k => k + 1);
            }, 150);
        });
        observer.observe(el);
        return () => {
            clearTimeout(timer);
            observer.disconnect();
        };
    }, []);

    const { nodes: layoutNodes, edges } = useMemo(() => {
        if (!data) return { nodes: [], edges: [] };
        return layoutWithDagre(data.nodes, data.edges);
    }, [data]);

    useEffect(() => {
        if (events.length <= processedEventCountRef.current) return;
        const newEvents = events.slice(processedEventCountRef.current);
        processedEventCountRef.current = events.length;

        setRunningNodeIds(prev => {
            const next = new Set(prev);
            for (const evt of newEvents) {
                if (evt.tool !== 'pipeline') continue;

                if (evt.status === 'running') {
                    if (evt.action === 'run_backtest') {
                        next.add('ensemble');
                    }
                    const result = typeof evt.result === 'string' ? evt.result : '';
                    for (const node of layoutNodes) {
                        if (node.type === 'model' && result.includes(String(node.data.label))) {
                            next.add(node.id);
                        }
                    }
                } else if (evt.status === 'success' || evt.status === 'error') {
                    if (evt.action === 'run_backtest') {
                        next.delete('ensemble');
                        for (const node of layoutNodes) {
                            if (node.type === 'model') next.delete(node.id);
                        }
                    }
                }
            }
            return next;
        });
    }, [events, layoutNodes]);

    const nodes = useMemo(() => {
        const selectedId = selectedNode?.id ?? null;
        if (runningNodeIds.size === 0 && !selectedId) return layoutNodes;
        return layoutNodes.map(n => {
            const extras: Record<string, unknown> = {};
            if (runningNodeIds.has(n.id)) extras._running = true;
            if (n.id === selectedId) extras._selected = true;
            if (Object.keys(extras).length === 0) return n;
            return { ...n, data: { ...n.data, ...extras } };
        });
    }, [layoutNodes, runningNodeIds, selectedNode]);

    const onNodeClick: NodeMouseHandler = useCallback((_event, node) => {
        setSelectedNode(node);
    }, []);

    const onPaneClick = useCallback(() => {
        setSelectedNode(null);
    }, []);

    if (loading) {
        return <div className={styles.loading}>Loading DAG...</div>;
    }

    if (error) {
        return <div className={styles.loading}>Error: {error}</div>;
    }

    return (
        <div className={styles.dagContainer}>
            <div className={styles.dagCanvas} ref={canvasRef}>
                <ReactFlow
                    key={resizeKey}
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    onNodeClick={onNodeClick}
                    onPaneClick={onPaneClick}
                    fitView
                    proOptions={{ hideAttribution: true }}
                >
                    <Background color="#30363d" gap={20} size={1} />
                    <Controls />
                    <MiniMap
                        nodeColor={miniMapNodeColor}
                        nodeStrokeWidth={3}
                        pannable
                        zoomable
                        style={{
                            backgroundColor: '#161b22',
                        }}
                    />
                </ReactFlow>
            </div>
            {selectedNode && (
                <DetailPanel
                    node={selectedNode}
                    onClose={() => setSelectedNode(null)}
                />
            )}
        </div>
    );
}
