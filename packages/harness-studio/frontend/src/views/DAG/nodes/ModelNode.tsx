import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface ModelNodeData {
    label: string;
    model_type?: string;
    features?: string[];
    active?: boolean;
    [key: string]: unknown;
}

export function ModelNode({ data }: { data: ModelNodeData }) {
    const isActive = data.active !== false;

    return (
        <div
            className={`${styles.node} ${styles.modelNode}`}
            style={{ opacity: isActive ? 1 : 0.5 }}
        >
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.model_type && (
                <div className={styles.nodeSubtext}>{data.model_type}</div>
            )}
            {data.features && (
                <div className={styles.nodeDetail}>{data.features.length} features</div>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </div>
    );
}
