import { useState } from 'react';
import { useApi } from '../../hooks/useApi';
import { useProject } from '../../hooks/useProject';
import styles from './Config.module.css';

interface ConfigFile {
    name: string;
    size: number;
}

interface ConfigContent {
    name: string;
    content: string;
}

function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    return `${(bytes / 1024).toFixed(1)}K`;
}

export function ConfigView() {
    const project = useProject();
    const { data: files, loading, error } = useApi<ConfigFile[]>('/api/config/files', undefined, project);
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const { data: content } = useApi<ConfigContent>(
        selectedFile ? `/api/config/files/${selectedFile}` : null,
        selectedFile,
        project,
    );

    if (loading) return <div className={styles.emptyState}>Loading config files...</div>;
    if (error) return <div className={styles.emptyState}>Error: {error}</div>;
    if (!files || files.length === 0) {
        return <div className={styles.emptyState}>No config files found.</div>;
    }

    return (
        <div className={styles.config}>
            <div className={styles.fileList}>
                {files.map(f => (
                    <button
                        key={f.name}
                        className={`${styles.fileItem} ${selectedFile === f.name ? styles.fileItemActive : ''}`}
                        onClick={() => setSelectedFile(f.name)}
                    >
                        {f.name}
                        <span className={styles.fileSize}>{formatSize(f.size)}</span>
                    </button>
                ))}
            </div>
            <div className={styles.viewer}>
                {content ? (
                    <>
                        <div className={styles.fileName}>{content.name}</div>
                        <pre className={styles.codeBlock}>{content.content}</pre>
                    </>
                ) : (
                    <div className={styles.emptyViewer}>
                        Select a config file to view
                    </div>
                )}
            </div>
        </div>
    );
}
