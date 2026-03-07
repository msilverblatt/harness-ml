import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface FeatureNodeData {
    label: string;
    count?: number;
    [key: string]: unknown;
}

export function FeatureNode({ data }: { data: FeatureNodeData }) {
    return (
        <div className={`${styles.node} ${styles.featuresNode}`}>
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.count != null && (
                <div className={styles.nodeSubtext}>{data.count} features</div>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </div>
    );
}
