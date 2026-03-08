import { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import { useTheme } from '../../hooks/useTheme';
import styles from './Features.module.css';

interface ImportanceEntry {
    name: string;
    importance: number;
    rank: number;
}

interface ImportanceResponse {
    features: ImportanceEntry[];
    source: 'model' | 'usage';
}

interface Feature {
    name: string;
    type: string;
    source: string | null;
    derived_from: string[];
    formula: string | null;
    category: string;
    enabled: boolean;
    used_by: string[];
}

export function FeaturesView() {
    const { colors } = useTheme();
    const project = useProject();
    const { data: features, loading, error } = useApi<Feature[]>('/api/features', undefined, project);
    const { data: importance } = useApi<ImportanceResponse>('/api/features/importance', undefined, project);
    const [search, setSearch] = useState('');
    const [typeFilter, setTypeFilter] = useState('all');
    const [sortCol, setSortCol] = useState<string>('name');
    const [sortAsc, setSortAsc] = useState(true);

    const types = useMemo(() => {
        if (!features) return [];
        return [...new Set(features.map(f => f.type))].sort();
    }, [features]);

    const filtered = useMemo(() => {
        if (!features) return [];
        let result = features;
        if (search) {
            const q = search.toLowerCase();
            result = result.filter(f =>
                f.name.toLowerCase().includes(q) ||
                (f.formula && f.formula.toLowerCase().includes(q)) ||
                f.used_by.some(m => m.toLowerCase().includes(q))
            );
        }
        if (typeFilter !== 'all') {
            result = result.filter(f => f.type === typeFilter);
        }
        result = [...result].sort((a, b) => {
            let cmp = 0;
            if (sortCol === 'name') cmp = a.name.localeCompare(b.name);
            else if (sortCol === 'type') cmp = a.type.localeCompare(b.type);
            else if (sortCol === 'used_by') cmp = a.used_by.length - b.used_by.length;
            return sortAsc ? cmp : -cmp;
        });
        return result;
    }, [features, search, typeFilter, sortCol, sortAsc]);

    const summary = useMemo(() => {
        if (!features) return null;
        const by_type: Record<string, number> = {};
        let used_count = 0;
        for (const f of features) {
            by_type[f.type] = (by_type[f.type] || 0) + 1;
            if (f.used_by.length > 0) used_count++;
        }
        return {
            total: features.length,
            used_by_models: used_count,
            unused: features.length - used_count,
            by_type,
        };
    }, [features]);

    const chartData = useMemo(() => {
        if (!importance || !importance.features || importance.features.length === 0) return null;
        return importance.features.slice(0, 20).map(f => ({
            name: f.name,
            importance: f.importance,
        }));
    }, [importance]);

    const handleSort = (col: string) => {
        if (sortCol === col) setSortAsc(!sortAsc);
        else { setSortCol(col); setSortAsc(true); }
    };

    if (loading) return <div className={styles.emptyState}>Loading features...</div>;
    if (error) return <div className={styles.emptyState}>Error: {error}</div>;

    return (
        <div className={styles.features}>
            {summary && (
                <div className={styles.statBar}>
                    <div className={styles.stat}>
                        <div className={styles.statValue}>{summary.total}</div>
                        <div className={styles.statLabel}>Total</div>
                    </div>
                    <div className={styles.stat}>
                        <div className={styles.statValue}>{summary.used_by_models}</div>
                        <div className={styles.statLabel}>Used</div>
                    </div>
                    <div className={styles.stat}>
                        <div className={styles.statValue}>{summary.unused}</div>
                        <div className={styles.statLabel}>Unused</div>
                    </div>
                    {Object.entries(summary.by_type).map(([type, count]) => (
                        <div className={styles.stat} key={type}>
                            <div className={styles.statValue}>{count}</div>
                            <div className={styles.statLabel}>{type}</div>
                        </div>
                    ))}
                </div>
            )}

            {chartData && (
                <div className={styles.chartSection}>
                    <div className={styles.sectionTitle}>
                        Feature Importance {importance && `(${importance.source})`}
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={chartData} layout="vertical" margin={{ left: 120, right: 20, top: 5, bottom: 5 }}>
                            <XAxis type="number" domain={[0, 1]} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                            <YAxis type="category" dataKey="name" width={110} tick={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                            <Tooltip
                                contentStyle={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)', fontSize: 12 }}
                                labelStyle={{ color: 'var(--color-text-primary)' }}
                                itemStyle={{ color: 'var(--color-accent)' }}
                            />
                            <Bar dataKey="importance" fill={colors.accent} radius={[0, 3, 3, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            <div className={styles.controls}>
                <input
                    className={styles.searchInput}
                    placeholder="Search features..."
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                />
                <select
                    className={styles.filterSelect}
                    value={typeFilter}
                    onChange={e => setTypeFilter(e.target.value)}
                >
                    <option value="all">All types</option>
                    {types.map(t => (
                        <option key={t} value={t}>{t}</option>
                    ))}
                </select>
            </div>

            {filtered.length === 0 ? (
                <div className={styles.emptyState}>No features match your filters.</div>
            ) : (
                <table className={styles.table}>
                    <thead>
                        <tr>
                            <th onClick={() => handleSort('name')}>
                                Name {sortCol === 'name' ? (sortAsc ? '↑' : '↓') : ''}
                            </th>
                            <th onClick={() => handleSort('type')}>
                                Type {sortCol === 'type' ? (sortAsc ? '↑' : '↓') : ''}
                            </th>
                            <th>Formula</th>
                            <th>Category</th>
                            <th onClick={() => handleSort('used_by')}>
                                Used By {sortCol === 'used_by' ? (sortAsc ? '↑' : '↓') : ''}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {filtered.map(f => (
                            <tr key={f.name} className={!f.enabled ? styles.disabled : ''}>
                                <td>{f.name}</td>
                                <td><span className={styles.typeBadge}>{f.type}</span></td>
                                <td><span className={styles.formula}>{f.formula || '--'}</span></td>
                                <td>{f.category}</td>
                                <td>
                                    {f.used_by.length > 0
                                        ? f.used_by.map(m => (
                                            <span key={m} className={styles.modelTag}>{m}</span>
                                        ))
                                        : <span style={{ color: 'var(--color-text-muted)' }}>unused</span>
                                    }
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}
