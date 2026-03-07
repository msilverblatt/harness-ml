import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface CalibrationNodeData {
    label: string;
    type?: string;
    [key: string]: unknown;
}

export function CalibrationNode({ data }: { data: CalibrationNodeData }) {
    return (
        <div className={`${styles.node} ${styles.calibrationNode}`}>
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.type && (
                <div className={styles.nodeSubtext}>{data.type}</div>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </div>
    );
}
