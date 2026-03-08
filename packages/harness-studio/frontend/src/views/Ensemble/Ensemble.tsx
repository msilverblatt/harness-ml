import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import { MetaCoefficients } from '../Diagnostics/MetaCoefficients';
import { CorrelationMatrix } from '../Diagnostics/CorrelationMatrix';
import { CalibrationPlot } from '../Diagnostics/CalibrationPlot';
import { Tooltip, MetricLabel } from '../../components/Tooltip/Tooltip';
import styles from './Ensemble.module.css';

interface EnsembleConfig {
    method: string;
    meta_learner: Record<string, unknown>;
    temperature: number | null;
    clip_floor: number | null;
    calibration: Record<string, unknown>;
    cv_strategy: string | null;
    fold_column: string | null;
    metrics: string[];
    exclude_models: string[];
    pre_calibration: Record<string, unknown>;
    task_type: string | null;
    target_column: string | null;
    model_count: number;
}

interface WeightsResponse {
    run_id: string;
    coefficients: Record<string, number>;
    pooled_metrics: Record<string, unknown>;
    task_type: string | null;
}

export function EnsembleView() {
    const project = useProject();
    const { data: config, loading: configLoading } = useApi<EnsembleConfig>(
        '/api/ensemble/config', undefined, project
    );
    const { data: weights, loading: weightsLoading } = useApi<WeightsResponse>(
        '/api/ensemble/weights', undefined, project
    );

    if (configLoading || weightsLoading) {
        return <div className={styles.emptyState}>Loading ensemble...</div>;
    }

    const runId = weights?.run_id;

    const taskType = config?.task_type || weights?.task_type || null;

    // Determine which metrics to display based on task type
    const getMetricEntries = (): Array<{ key: string; label: string; value: string }> => {
        const pm = weights?.pooled_metrics;
        if (!pm) return [];

        const fmt = (v: unknown): string => {
            if (v == null) return '--';
            const n = Number(v);
            if (isNaN(n)) return String(v);
            return n.toFixed(4);
        };

        if (taskType === 'binary') {
            const keys = ['accuracy', 'brier', 'ece', 'log_loss', 'auc'];
            return keys
                .filter(k => pm[k] != null)
                .map(k => ({ key: k, label: k.toUpperCase(), value: fmt(pm[k]) }));
        }
        if (taskType === 'regression') {
            const keys = ['rmse', 'mae', 'r2', 'mape'];
            return keys
                .filter(k => pm[k] != null)
                .map(k => ({ key: k, label: k.toUpperCase(), value: fmt(pm[k]) }));
        }
        // For any other task type, show all numeric metrics
        return Object.entries(pm)
            .filter(([, v]) => typeof v === 'number')
            .map(([k, v]) => ({ key: k, label: k.toUpperCase(), value: fmt(v) }));
    };

    const metricEntries = getMetricEntries();

    return (
        <div className={styles.ensemble}>
            {config && (
                <div className={styles.configSection}>
                    <div className={styles.configGrid}>
                        <div className={styles.configItem}>
                            <span className={styles.configValue}><Tooltip term={config.method} label={config.method} /></span>
                            <span className={styles.configLabel}>Method</span>
                        </div>
                        {taskType && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}>{taskType}</span>
                                <span className={styles.configLabel}>Task Type</span>
                            </div>
                        )}
                        {config.target_column && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}>{config.target_column}</span>
                                <span className={styles.configLabel}>Target</span>
                            </div>
                        )}
                        {config.model_count > 0 && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}>{config.model_count}</span>
                                <span className={styles.configLabel}>Models</span>
                            </div>
                        )}
                        {config.cv_strategy && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}><Tooltip term="cv_strategy" label={config.cv_strategy} /></span>
                                <span className={styles.configLabel}>CV Strategy</span>
                            </div>
                        )}
                        {config.fold_column && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}><Tooltip term="fold_column" label={config.fold_column} /></span>
                                <span className={styles.configLabel}>Fold Column</span>
                            </div>
                        )}
                        {config.temperature != null && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}><Tooltip term="temperature" label={String(config.temperature)} /></span>
                                <span className={styles.configLabel}>Temperature</span>
                            </div>
                        )}
                        {config.clip_floor != null && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}><Tooltip term="clip_floor" label={String(config.clip_floor)} /></span>
                                <span className={styles.configLabel}>Clip Floor</span>
                            </div>
                        )}
                        {config.calibration && Object.keys(config.calibration).length > 0 && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}>
                                    {typeof config.calibration === 'object'
                                        ? (config.calibration as Record<string, unknown>).type as string || 'configured'
                                        : String(config.calibration)}
                                </span>
                                <span className={styles.configLabel}>Calibration</span>
                            </div>
                        )}
                        {config.metrics.length > 0 && (
                            <div className={styles.configItem}>
                                <span className={styles.configValue}>{config.metrics.join(', ')}</span>
                                <span className={styles.configLabel}>Metrics</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {metricEntries.length > 0 && (
                <div className={styles.configSection}>
                    <div className={styles.configGrid}>
                        {metricEntries.map(m => (
                            <div key={m.key} className={styles.configItem}>
                                <span className={styles.configValue}>{m.value}</span>
                                <span className={styles.configLabel}><MetricLabel name={m.key} /></span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {runId && (
                <>
                    <div className={styles.panel}>
                        <div className={styles.sectionTitle}><Tooltip term="meta_coefficients" label="Model Weights" /></div>
                        <MetaCoefficients runId={runId} />
                    </div>

                    <div className={styles.twoColumn}>
                        <div className={styles.panel}>
                            <div className={styles.sectionTitle}><Tooltip term="correlation" label="Correlation Matrix" /></div>
                            <CorrelationMatrix runId={runId} />
                        </div>
                        <div className={styles.panel}>
                            <div className={styles.sectionTitle}><Tooltip term="calibration" label="Calibration" /></div>
                            <CalibrationPlot runId={runId} />
                        </div>
                    </div>
                </>
            )}

            {!runId && !configLoading && (
                <div className={styles.emptyState}>No runs available for ensemble analysis.</div>
            )}
        </div>
    );
}
