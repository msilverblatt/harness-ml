import { Handle, Position } from '@xyflow/react';
import styles from './nodeStyles.module.css';

interface EnsembleNodeData {
    label: string;
    method?: string;
    model_count?: number;
    cv_strategy?: string;
    metrics?: string[];
    fold_column?: string;
    [key: string]: unknown;
}

export function EnsembleNode({ data }: { data: EnsembleNodeData }) {
    const metrics = data.metrics ?? [];
    const isRunning = !!(data as Record<string, unknown>)._running;
    return (
        <div className={`${styles.node} ${styles.ensembleNode}${isRunning ? ` ${styles.nodeRunning}` : ''}`}>
            <div className={styles.nodeLabel}>
                {data.label}
                {isRunning && <span className={styles.runningIndicator}>running</span>}
            </div>
            <div className={styles.nodeSubtext}>
                {data.method ?? 'stacking'}
                {data.model_count != null ? ` · ${data.model_count} models` : ''}
            </div>
            {data.cv_strategy && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeParamRow}>
                        <span className={styles.nodeParamKey}>cv</span>
                        <span className={styles.nodeParamValue}>{data.cv_strategy}</span>
                    </div>
                    {data.fold_column && (
                        <div className={styles.nodeParamRow}>
                            <span className={styles.nodeParamKey}>fold</span>
                            <span className={styles.nodeParamValue}>{data.fold_column}</span>
                        </div>
                    )}
                </>
            )}
            {metrics.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeDetail}>metrics</div>
                    <div className={styles.nodeDetailList}>
                        {metrics.map(m => (
                            <span key={m} className={styles.nodeTag}>{m}</span>
                        ))}
                    </div>
                </>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </div>
    );
}
