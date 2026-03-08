import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
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
    const project = useProject();
    const { data, loading, error } = useApi<ReportResponse>(`/api/runs/${runId}/report`, undefined, project);

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
