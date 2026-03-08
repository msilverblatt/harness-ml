import { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';

interface ProjectInfo {
    name: string;
    project_dir: string;
    last_seen: number;
}

export function ProjectRedirect() {
    const [project, setProject] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/api/projects')
            .then(r => r.json())
            .then((projects: ProjectInfo[]) => {
                if (projects.length > 0) {
                    // Already sorted by last_seen DESC from backend
                    setProject(projects[0].name);
                }
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, []);

    if (loading) {
        return <div style={{ padding: '2rem', color: 'var(--c-text-2)' }}>Loading...</div>;
    }

    if (project) {
        return <Navigate to={`/${project}/dashboard`} replace />;
    }

    return (
        <div style={{ padding: '2rem', color: 'var(--c-text-2)' }}>
            No projects registered yet. Run <code>configure(action="init")</code> to get started.
        </div>
    );
}
