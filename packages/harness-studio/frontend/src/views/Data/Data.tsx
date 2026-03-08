import { useState } from 'react';
import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import styles from './Data.module.css';

interface Source {
    name: string;
    path: string;
    columns: string[];
    rows: number;
    adapter: string;
    is_bootstrap: boolean;
}

interface ColumnStat {
    name: string;
    dtype: string;
    null_count: number;
    null_pct: number;
    unique_count: number;
    min?: number | null;
    max?: number | null;
    mean?: number | null;
}

interface StatsResponse {
    columns: ColumnStat[];
    total_rows: number;
}

interface PreviewResponse {
    columns: string[];
    dtypes: Record<string, string>;
    rows: Record<string, unknown>[];
    total_rows: number;
}

function SourceCard({ source, initialExpanded = false }: { source: Source; initialExpanded?: boolean }) {
    const project = useProject();
    const [expanded, setExpanded] = useState(initialExpanded);
    const { data: stats } = useApi<StatsResponse>(
        expanded ? `/api/data/sources/${source.name}/stats` : null,
        undefined,
        project,
    );
    const { data: preview } = useApi<PreviewResponse>(
        expanded ? `/api/data/sources/${source.name}/preview?rows=20` : null,
        undefined,
        project,
    );

    return (
        <div className={styles.sourceCard}>
            <div className={styles.sourceHeader} onClick={() => setExpanded(!expanded)}>
                <span className={`${styles.chevron} ${expanded ? styles.chevronOpen : ''}`}>&#9656;</span>
                <span className={styles.sourceName}>{source.name}</span>
                <span className={styles.adapterBadge}>{source.adapter}</span>
                <span className={styles.sourceMeta}>
                    <span>{source.rows.toLocaleString()} rows</span>
                    <span>{source.columns.length} cols</span>
                </span>
            </div>
            {expanded && (
                <div className={styles.sourceBody}>
                    <div>
                        <div className={styles.sectionLabel}>Path</div>
                        <code style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
                            {source.path}
                        </code>
                    </div>

                    {stats && 'columns' in stats && (
                        <div>
                            <div className={styles.sectionLabel}>Column Statistics</div>
                            <table className={styles.statsTable}>
                                <thead>
                                    <tr>
                                        <th>Column</th>
                                        <th>Type</th>
                                        <th>Nulls</th>
                                        <th>Unique</th>
                                        <th>Min</th>
                                        <th>Max</th>
                                        <th>Mean</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stats.columns.map(col => (
                                        <tr key={col.name}>
                                            <td>{col.name}</td>
                                            <td>{col.dtype}</td>
                                            <td className={col.null_count > 0 ? styles.nullHighlight : ''}>
                                                {col.null_count} ({col.null_pct}%)
                                            </td>
                                            <td>{col.unique_count}</td>
                                            <td>{col.min != null ? col.min : '--'}</td>
                                            <td>{col.max != null ? col.max : '--'}</td>
                                            <td>{col.mean != null ? col.mean : '--'}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {preview && 'rows' in preview && preview.rows.length > 0 && (
                        <div>
                            <div className={styles.sectionLabel}>Preview (first {preview.rows.length} rows)</div>
                            <div className={styles.previewWrap}>
                                <table className={styles.previewTable}>
                                    <thead>
                                        <tr>
                                            {preview.columns.map(col => (
                                                <th key={col}>{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {preview.rows.map((row, i) => (
                                            <tr key={i}>
                                                {preview.columns.map(col => (
                                                    <td key={col}>{String(row[col] ?? '')}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export function DataView() {
    const project = useProject();
    const { data: sources, loading, error } = useApi<Source[]>('/api/data/sources', undefined, project);

    if (loading) return <div className={styles.emptyState}>Loading data sources...</div>;
    if (error) return <div className={styles.emptyState}>Error: {error}</div>;
    if (!sources || sources.length === 0) {
        return <div className={styles.emptyState}>No data sources registered.</div>;
    }

    const totalRows = sources.reduce((sum, s) => sum + s.rows, 0);
    const totalCols = sources.reduce((sum, s) => sum + s.columns.length, 0);

    return (
        <div className={styles.data}>
            <div className={styles.statBar}>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{sources.length}</span>
                    <span className={styles.statLabel}>Sources</span>
                </div>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{totalRows.toLocaleString()}</span>
                    <span className={styles.statLabel}>Total Rows</span>
                </div>
                <div className={styles.stat}>
                    <span className={styles.statValue}>{totalCols.toLocaleString()}</span>
                    <span className={styles.statLabel}>Total Columns</span>
                </div>
            </div>
            {sources.map((src, i) => (
                <SourceCard key={src.name} source={src} initialExpanded={i === 0} />
            ))}
        </div>
    );
}
