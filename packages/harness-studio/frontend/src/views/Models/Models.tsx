import { useState, useMemo } from 'react';
import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import { Tooltip, MetricLabel } from '../../components/Tooltip/Tooltip';
import styles from './Models.module.css';

interface Model {
    name: string;
    type: string;
    mode: string | null;
    params: Record<string, unknown>;
    features: string[];
    feature_count: number;
    active: boolean;
    include_in_ensemble: boolean;
    source: string;
}

interface ModelMetrics {
    model: string;
    run_id: string;
    metrics: Record<string, number>;
    n_predictions: number;
}

function ModelRow({ model }: { model: Model }) {
    const project = useProject();
    const [expanded, setExpanded] = useState(false);
    const { data: metricsData } = useApi<ModelMetrics>(
        expanded ? `/api/models/${model.name}/metrics` : null,
        undefined,
        project,
    );

    const statusClass = model.source === 'experimental'
        ? styles.statusExperimental
        : model.active ? styles.statusActive : styles.statusInactive;
    const statusLabel = model.source === 'experimental'
        ? 'experimental'
        : model.active ? 'active' : 'inactive';

    return (
        <>
            <tr className={styles.clickableRow} onClick={() => setExpanded(!expanded)}>
                <td>
                    <span className={`${styles.chevron} ${expanded ? styles.chevronOpen : ''}`}>&#9656;</span>
                    <span className={styles.modelName}>{model.name}</span>
                </td>
                <td><span className={styles.typeBadge}><Tooltip term={model.type} label={model.type} /></span></td>
                <td><span className={styles.muted}>{model.mode || '--'}</span></td>
                <td>{model.feature_count}</td>
                <td><span className={`${styles.statusBadge} ${statusClass}`}>{statusLabel}</span></td>
                <td><span className={styles.muted}>{model.source}</span></td>
            </tr>
            {expanded && (
                <tr className={styles.detailRow}>
                    <td colSpan={6}>
                        <div className={styles.detailBody}>
                            {metricsData && metricsData.metrics && (
                                <div>
                                    <div className={styles.sectionLabel}>Metrics (latest run)</div>
                                    <div className={styles.metricsRow}>
                                        {Object.entries(metricsData.metrics).map(([k, v]) => (
                                            <div key={k}>
                                                <span className={styles.metricLabel}><MetricLabel name={k} />: </span>
                                                <span className={styles.metricValue}>{v}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div>
                                <div className={styles.sectionLabel}>Features</div>
                                <div>
                                    {model.features.map(f => (
                                        <span key={f} className={styles.featureTag}>{f}</span>
                                    ))}
                                </div>
                            </div>

                            {Object.keys(model.params).length > 0 && (
                                <div>
                                    <div className={styles.sectionLabel}>Parameters</div>
                                    {Object.entries(model.params).map(([k, v]) => (
                                        <div key={k} className={styles.paramRow}>
                                            <span className={styles.paramKey}><Tooltip term={k} label={k} /></span>
                                            <span className={styles.paramValue}>{String(v)}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </td>
                </tr>
            )}
        </>
    );
}

export function ModelsView() {
    const project = useProject();
    const { data: models, loading, error } = useApi<Model[]>('/api/models', undefined, project);

    const summary = useMemo(() => {
        if (!models) return null;
        const by_type: Record<string, number> = {};
        let active_count = 0;
        for (const m of models) {
            by_type[m.type] = (by_type[m.type] || 0) + 1;
            if (m.active) active_count++;
        }
        return {
            total: models.length,
            active: active_count,
            by_type,
        };
    }, [models]);

    if (loading) return <div className={styles.emptyState}>Loading models...</div>;
    if (error) return <div className={styles.emptyState}>Error: {error}</div>;
    if (!models || models.length === 0) {
        return <div className={styles.emptyState}>No models configured.</div>;
    }

    return (
        <div className={styles.models}>
            {summary && (
                <div className={styles.statBar}>
                    <div className={styles.stat}>
                        <div className={styles.statValue}>{summary.total}</div>
                        <div className={styles.statLabel}>Total</div>
                    </div>
                    <div className={styles.stat}>
                        <div className={styles.statValue}>{summary.active}</div>
                        <div className={styles.statLabel}>Active</div>
                    </div>
                    {Object.entries(summary.by_type).map(([type, count]) => (
                        <div className={styles.stat} key={type}>
                            <div className={styles.statValue}>{count}</div>
                            <div className={styles.statLabel}>{type}</div>
                        </div>
                    ))}
                </div>
            )}

            <table className={styles.table}>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Mode</th>
                        <th>Features</th>
                        <th>Status</th>
                        <th>Source</th>
                    </tr>
                </thead>
                <tbody>
                    {models.map(m => (
                        <ModelRow key={m.name} model={m} />
                    ))}
                </tbody>
            </table>
        </div>
    );
}
