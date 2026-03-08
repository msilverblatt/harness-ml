import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import { Tooltip, MetricLabel } from '../../../components/Tooltip/Tooltip';
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
    const isRunning = !!data._running;
    return (
        <NodeShell typeClass={styles.ensembleNode} data={data}>
            <div className={styles.nodeLabel}>
                {data.label}
                {isRunning && <span className={styles.runningIndicator}>running</span>}
            </div>
            <div className={styles.nodeSubtext}>
                <Tooltip term={data.method ?? 'stacking'} label={data.method ?? 'stacking'} />
                {data.model_count != null ? ` · ${data.model_count} models` : ''}
            </div>
            {data.cv_strategy && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeParamRow}>
                        <span className={styles.nodeParamKey}><Tooltip term="cv_strategy" label="cv" /></span>
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
                            <span key={m} className={styles.nodeTag}><MetricLabel name={m} /></span>
                        ))}
                    </div>
                </>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </NodeShell>
    );
}
