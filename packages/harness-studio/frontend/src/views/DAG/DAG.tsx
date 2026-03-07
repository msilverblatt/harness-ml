import { useMemo } from 'react';
import { ReactFlow, Background, Controls, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';
import { useApi } from '../../hooks/useApi';
import { SourceNode } from './nodes/SourceNode';
import { FeatureNode } from './nodes/FeatureNode';
import { ModelNode } from './nodes/ModelNode';
import { EnsembleNode } from './nodes/EnsembleNode';
import { CalibrationNode } from './nodes/CalibrationNode';
import { OutputNode } from './nodes/OutputNode';
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
};

function layoutWithDagre(apiNodes: ApiNode[], apiEdges: ApiEdge[]): { nodes: Node[]; edges: Edge[] } {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'LR', nodesep: 50, ranksep: 100 });

    apiNodes.forEach(n => g.setNode(n.id, { width: 160, height: 70 }));
    apiEdges.forEach(e => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const nodes: Node[] = apiNodes.map(n => {
        const pos = g.node(n.id);
        return {
            id: n.id,
            type: n.type,
            position: { x: pos.x - 80, y: pos.y - 35 },
            data: { label: n.label, ...n.data },
        };
    });

    const edges: Edge[] = apiEdges.map((e, i) => ({
        id: `e${i}`,
        source: e.source,
        target: e.target,
        style: { stroke: '#30363d' },
    }));

    return { nodes, edges };
}

export function DAG() {
    const { data, loading, error } = useApi<DagResponse>('/api/project/dag');

    const { nodes, edges } = useMemo(() => {
        if (!data) return { nodes: [], edges: [] };
        return layoutWithDagre(data.nodes, data.edges);
    }, [data]);

    if (loading) {
        return <div className={styles.loading}>Loading DAG...</div>;
    }

    if (error) {
        return <div className={styles.loading}>Error: {error}</div>;
    }

    return (
        <div className={styles.dagContainer}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                fitView
                proOptions={{ hideAttribution: true }}
            >
                <Background color="#30363d" gap={20} size={1} />
                <Controls />
            </ReactFlow>
        </div>
    );
}
