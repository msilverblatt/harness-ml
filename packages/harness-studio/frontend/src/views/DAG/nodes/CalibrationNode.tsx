import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import { Tooltip } from '../../../components/Tooltip/Tooltip';
import styles from './nodeStyles.module.css';

interface CalibrationNodeData {
    label: string;
    type?: string;
    [key: string]: unknown;
}

export function CalibrationNode({ data }: { data: CalibrationNodeData }) {
    return (
        <NodeShell typeClass={styles.calibrationNode} data={data}>
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.type && (
                <div className={styles.nodeSubtext}><Tooltip term={data.type} label={data.type} /></div>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </NodeShell>
    );
}
