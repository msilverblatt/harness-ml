import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import styles from './nodeStyles.module.css';

interface SourceNodeData {
    label: string;
    path?: string;
    columns?: string[];
    rows?: number;
    adapter?: string;
    [key: string]: unknown;
}

export function SourceNode({ data }: { data: SourceNodeData }) {
    const columns = data.columns ?? [];
    return (
        <NodeShell typeClass={styles.sourceNode} data={data}>
            <div className={styles.nodeLabel}>{data.label}</div>
            {data.path && (
                <div className={styles.nodeDetail}>{data.path}</div>
            )}
            {data.rows != null && data.rows > 0 && (
                <div className={styles.nodeSubtext}>{data.rows.toLocaleString()} rows</div>
            )}
            {columns.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeDetail}>{columns.length} columns</div>
                    <div className={styles.nodeDetailList}>
                        {columns.slice(0, 8).map(c => (
                            <span key={c} className={styles.nodeTag}>{c}</span>
                        ))}
                        {columns.length > 8 && (
                            <span className={styles.nodeTag}>+{columns.length - 8} more</span>
                        )}
                    </div>
                </>
            )}
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </NodeShell>
    );
}
