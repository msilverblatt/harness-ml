import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import styles from './nodeStyles.module.css';

interface ViewNodeData {
    label: string;
    steps?: string[];
    step_count?: number;
    [key: string]: unknown;
}

export function ViewNode({ data }: { data: ViewNodeData }) {
    const steps = data.steps ?? [];
    return (
        <NodeShell typeClass={styles.viewNode} data={data}>
            <div className={styles.nodeLabel}>{data.label}</div>
            <div className={styles.nodeSubtext}>{data.step_count ?? steps.length} transforms</div>
            {steps.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeDetailList}>
                        {steps.map((s, i) => (
                            <span key={i} className={styles.nodeTag}>{s}</span>
                        ))}
                    </div>
                </>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </NodeShell>
    );
}
