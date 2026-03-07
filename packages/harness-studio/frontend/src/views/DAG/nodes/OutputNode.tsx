import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface OutputNodeData {
    label: string;
    [key: string]: unknown;
}

export function OutputNode({ data }: { data: OutputNodeData }) {
    return (
        <div className={`${styles.node} ${styles.outputNode}`}>
            <div className={styles.nodeLabel}>{data.label}</div>
            <Handle type="target" position={Position.Left} className={styles.handle} />
        </div>
    );
}
