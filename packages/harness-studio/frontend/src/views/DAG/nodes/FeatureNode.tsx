import { Handle, Position } from '@xyflow/react';
import { NodeShell } from './NodeShell';
import styles from './nodeStyles.module.css';

interface EngineeredFeature {
    name: string;
    formula: string;
}

interface FeatureNodeData {
    label: string;
    count?: number;
    features?: string[];
    raw_features?: string[];
    engineered_features?: EngineeredFeature[];
    target_column?: string;
    task?: string;
    [key: string]: unknown;
}

export function FeatureNode({ data }: { data: FeatureNodeData }) {
    const raw = data.raw_features ?? [];
    const engineered = data.engineered_features ?? [];

    return (
        <NodeShell typeClass={styles.featuresNode} data={data}>
            <div className={styles.nodeLabel}>{data.label}</div>
            <div className={styles.nodeSubtext}>
                {data.count ?? 0} features
                {data.task ? ` · ${data.task}` : ''}
            </div>
            {data.target_column && (
                <div className={styles.nodeDetail}>target: {data.target_column}</div>
            )}

            {raw.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeDetail}>{raw.length} raw</div>
                    <div className={styles.nodeDetailList}>
                        {raw.slice(0, 8).map(f => (
                            <span key={f} className={styles.nodeTag}>{f}</span>
                        ))}
                        {raw.length > 8 && (
                            <span className={styles.nodeTag}>+{raw.length - 8} more</span>
                        )}
                    </div>
                </>
            )}

            {engineered.length > 0 && (
                <>
                    <hr className={styles.nodeDivider} />
                    <div className={styles.nodeDetail}>{engineered.length} engineered</div>
                    <div className={styles.nodeDetailList}>
                        {engineered.slice(0, 6).map(f => (
                            <div key={f.name} className={styles.nodeParamRow}>
                                <span className={styles.nodeParamKey}>{f.name}</span>
                                <span className={styles.nodeParamValue}>{f.formula}</span>
                            </div>
                        ))}
                        {engineered.length > 6 && (
                            <div className={styles.nodeDetail}>+{engineered.length - 6} more</div>
                        )}
                    </div>
                </>
            )}
            <Handle type="target" position={Position.Left} className={styles.handle} />
            <Handle type="source" position={Position.Right} className={styles.handle} />
        </NodeShell>
    );
}
