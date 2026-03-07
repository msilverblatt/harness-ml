import { useApi } from '../../hooks/useApi';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import styles from './Diagnostics.module.css';

interface ReportResponse {
    markdown?: string;
    error?: string;
}

interface ReportViewProps {
    runId: string;
}

export function ReportView({ runId }: ReportViewProps) {
    const { data, loading, error } = useApi<ReportResponse>(`/api/runs/${runId}/report`);

    if (loading) {
        return <div className={styles.emptyState}>Loading report...</div>;
    }

    if (error || !data || data.error || !data.markdown) {
        return (
            <div className={styles.emptyState}>
                {data?.error ?? error ?? 'No report available for this run.'}
            </div>
        );
    }

    return (
        <div className={styles.reportContainer}>
            <MarkdownRenderer content={data.markdown} />
        </div>
    );
}
