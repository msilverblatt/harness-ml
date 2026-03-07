import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface SourceNodeData {
    label: string;
    [key: string]: unknown;
}

export function SourceNode({ data }: { data: SourceNodeData }) {
    return (
        <div className={`${styles.node} ${styles.sourceNode}`}>
            <div className={styles.nodeLabel}>{data.label}</div>
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </div>
    );
}
