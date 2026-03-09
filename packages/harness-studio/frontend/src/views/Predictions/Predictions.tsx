import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import { useTheme } from '../../hooks/useTheme';
import styles from './Predictions.module.css';

interface PredSummary {
    run_id: string;
    total_predictions: number;
    model_columns: string[];
    correct?: number;
    accuracy?: number;
    avg_confidence?: number;
    fold_counts?: Record<string, number>;
}

interface DistResponse {
    run_id: string;
    prob_column: string;
    histogram: { bin_center: number; count: number }[];
    stats: { mean: number; std: number; median: number };
    error?: string;
}

interface PredPage {
    run_id: string;
    columns: string[];
    rows: Record<string, unknown>[];
    total: number;
    page: number;
    page_size: number;
}

function getBaseUrl(): string {
    if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
        return import.meta.env.VITE_API_URL;
    }
    return window.location.origin;
}

function ExportButtons({ project }: { project: string }) {
    const handleExport = (format: 'csv' | 'parquet') => {
        let url = `${getBaseUrl()}/api/predictions/export?format=${format}`;
        if (project) {
            url += `&project=${encodeURIComponent(project)}`;
        }
        window.open(url, '_blank');
    };

    return (
        <div className={styles.exportButtons}>
            <button className={styles.exportBtn} onClick={() => handleExport('csv')}>
                Export CSV
            </button>
            <button className={styles.exportBtn} onClick={() => handleExport('parquet')}>
                Export Parquet
            </button>
        </div>
    );
}

export function PredictionsView() {
    const { colors } = useTheme();
    const project = useProject();
    const [page, setPage] = useState(0);
    const { data: summary, loading: summaryLoading } = useApi<PredSummary>(
        '/api/predictions/summary', undefined, project
    );
    const { data: dist } = useApi<DistResponse>(
        '/api/predictions/distribution', undefined, project
    );
    const { data: preds } = useApi<PredPage>(
        `/api/predictions?page=${page}&page_size=50`, page, project
    );

    if (summaryLoading) return <div className={styles.emptyState}>Loading predictions...</div>;

    return (
        <div className={styles.predictions}>
            {summary && (
                <>
                    <div className={styles.summaryBar}>
                        <div className={styles.summaryCard}>
                            <div className={styles.summaryValue}>{summary.total_predictions.toLocaleString()}</div>
                            <div className={styles.summaryLabel}>Total</div>
                        </div>
                        {summary.accuracy != null && (
                            <div className={styles.summaryCard}>
                                <div className={styles.summaryValue}>{(summary.accuracy * 100).toFixed(1)}%</div>
                                <div className={styles.summaryLabel}>Accuracy</div>
                            </div>
                        )}
                        {summary.avg_confidence != null && (
                            <div className={styles.summaryCard}>
                                <div className={styles.summaryValue}>{(summary.avg_confidence * 100).toFixed(1)}%</div>
                                <div className={styles.summaryLabel}>Avg Confidence</div>
                            </div>
                        )}
                        <div className={styles.summaryCard}>
                            <div className={styles.summaryValue}>{summary.model_columns.length}</div>
                            <div className={styles.summaryLabel}>Models</div>
                        </div>
                    </div>
                    <ExportButtons project={project} />
                </>
            )}

            {dist && dist.histogram && (
                <div className={styles.chartSection}>
                    <div className={styles.sectionTitle}>
                        Probability Distribution ({dist.prob_column})
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={dist.histogram}>
                            <XAxis
                                dataKey="bin_center"
                                tick={{ fill: colors.textSecondary, fontSize: 10 }}
                                tickFormatter={v => v.toFixed(2)}
                            />
                            <YAxis tick={{ fill: colors.textSecondary, fontSize: 10 }} />
                            <Tooltip
                                contentStyle={{
                                    background: colors.bgSecondary,
                                    border: `1px solid ${colors.border}`,
                                    borderRadius: 6,
                                    fontSize: 12,
                                }}
                            />
                            <Bar dataKey="count" fill={colors.accent} radius={[2, 2, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}
            {dist && !dist.histogram && dist.error && (
                <div className={styles.chartSection}>
                    <div className={styles.sectionTitle}>Probability Distribution</div>
                    <div className={styles.emptyState}>{dist.error}</div>
                </div>
            )}

            {preds && (
                <div className={styles.tableWrap}>
                    <div className={styles.tableScroll}>
                        <table className={styles.table}>
                            <thead>
                                <tr>
                                    {preds.columns.map(col => (
                                        <th key={col}>{col}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {preds.rows.map((row, i) => (
                                    <tr key={i}>
                                        {preds.columns.map(col => (
                                            <td key={col}>{String(row[col] ?? '')}</td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className={styles.pagination}>
                        <button
                            className={styles.pageBtn}
                            onClick={() => setPage(p => p - 1)}
                            disabled={page === 0}
                        >
                            Prev
                        </button>
                        <span>
                            {preds.page * preds.page_size + 1}–{Math.min((preds.page + 1) * preds.page_size, preds.total)} of {preds.total}
                        </span>
                        <button
                            className={styles.pageBtn}
                            onClick={() => setPage(p => p + 1)}
                            disabled={(page + 1) * 50 >= preds.total}
                        >
                            Next
                        </button>
                    </div>
                </div>
            )}

            {!summary && !summaryLoading && (
                <div className={styles.emptyState}>No predictions available.</div>
            )}
        </div>
    );
}
