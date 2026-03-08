import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import styles from './nodeStyles.module.css';

interface OutputNodeData {
    label: string;
    target?: string;
    [key: string]: unknown;
}

export function OutputNode({ data }: { data: OutputNodeData }) {
    return (
        <NodeShell typeClass={styles.outputNode} data={data}>
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.target && (
                <div className={styles.nodeSubtext}>target: {data.target}</div>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
        </NodeShell>
    );
}
