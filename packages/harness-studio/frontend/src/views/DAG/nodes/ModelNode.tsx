import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import { Tooltip } from '../../../components/Tooltip/Tooltip';
import styles from './nodeStyles.module.css';

interface ModelNodeData {
    label: string;
    model_type?: string;
    mode?: string;
    features?: string[];
    feature_count?: number;
    active?: boolean;
    params?: Record<string, unknown>;
    [key: string]: unknown;
}

const KEY_PARAMS = ['max_depth', 'learning_rate', 'n_estimators', 'num_leaves', 'C', 'max_iter', 'dropout', 'hidden_layers'];

export function ModelNode({ data }: { data: ModelNodeData }) {
    const isActive = data.active !== false;
    const isRunning = !!data._running;
    const isExperiment = !!data._experiment;
    const isNewExperiment = !!data._experiment_new;
    const features = data.features ?? [];
    const params = data.params ?? {};
    const displayParams = KEY_PARAMS.filter(k => k in params);

    const typeClass = isExperiment ? `${styles.modelNode} ${styles.experimentNode}` : styles.modelNode;

    return (
        <NodeShell typeClass={typeClass} data={data} style={{ opacity: isActive ? 1 : 0.5 }}>
            <div className={styles.nodeLabel}>
                {data.label}
                {isRunning && <span className={styles.runningIndicator}>running</span>}
                {isExperiment && (
                    <span className={styles.experimentBadge}>
                        {isNewExperiment ? 'NEW' : 'EXP'}
                    </span>
                )}
            </div>
            <div className={styles.nodeSubtext}>
                <Tooltip term={data.model_type ?? 'unknown'} label={data.model_type ?? 'unknown'} />
                {data.mode ? ` · ${data.mode}` : ''}
            </div>
            <div className={styles.nodeDetail}>{data.feature_count ?? features.length} features</div>
            {displayParams.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    {displayParams.slice(0, 5).map(k => (
                        <div key={k} className={styles.nodeParamRow}>
                            <span className={styles.nodeParamKey}><Tooltip term={k} label={k} /></span>
                            <span className={styles.nodeParamValue}>{String(params[k])}</span>
                        </div>
                    ))}
                </>
            )}
            {features.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeDetailList}>
                        {features.slice(0, 6).map(f => (
                            <span key={f} className={styles.nodeTag}>{f}</span>
                        ))}
                        {features.length > 6 && (
                            <span className={styles.nodeTag}>+{features.length - 6} more</span>
                        )}
                    </div>
                </>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </NodeShell>
    );
}
