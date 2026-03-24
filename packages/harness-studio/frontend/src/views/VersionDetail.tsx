import { useEffect, useState } from 'react';
import { api } from '../api';

export default function VersionDetail({ selectedVersion }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [detail, setDetail] = useState<any>(null);
  const [diagnostics, setDiagnostics] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedVersion) return;
    setDetail(null);
    setDiagnostics(null);
    setError(null);
    api.versions.detail(selectedVersion).then(setDetail).catch(e => setError(e.message));
    api.diagnostics(selectedVersion).then(setDiagnostics).catch(() => {});
  }, [selectedVersion]);

  if (!selectedVersion) return <div style={{ color: '#888' }}>Select a version from the Version Tree tab.</div>;
  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!detail) return <div>Loading...</div>;

  return (
    <div>
      <h2>Version: {detail.id}</h2>
      <table style={{ borderCollapse: 'collapse', marginBottom: 24 }}>
        <tbody>
          {Object.entries(detail).map(([key, val]) => (
            <tr key={key} style={{ borderBottom: '1px solid #333' }}>
              <td style={{ padding: 8, fontWeight: 'bold', verticalAlign: 'top' }}>{key}</td>
              <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12, whiteSpace: 'pre-wrap', maxWidth: 600 }}>
                {typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val ?? '-')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {diagnostics && (
        <>
          <h3>Diagnostics</h3>
          <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
            {JSON.stringify(diagnostics, null, 2)}
          </pre>
        </>
      )}
    </div>
  );
}
