import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface EnsembleNodeData {
    label: string;
    method?: string;
    [key: string]: unknown;
}

export function EnsembleNode({ data }: { data: EnsembleNodeData }) {
    return (
        <div className={`${styles.node} ${styles.ensembleNode}`}>
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.method && (
                <div className={styles.nodeSubtext}>{data.method}</div>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </div>
    );
}
