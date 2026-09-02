import { useEffect, useState } from 'react';
import { api } from '../api';

export default function Diagnostics({ selectedVersion }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedVersion) return;
    setData(null);
    setError(null);
    api.diagnostics(selectedVersion).then(setData).catch(e => setError(e.message));
  }, [selectedVersion]);

  if (!selectedVersion) return <div style={{ color: '#888' }}>Select a version from the Version Tree tab to view diagnostics.</div>;
  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!data) return <div>Loading...</div>;

  const metrics = data.metrics || data;
  const metricEntries = typeof metrics === 'object' && !Array.isArray(metrics) ? Object.entries(metrics) : [];

  return (
    <div>
      <h2>Diagnostics: {selectedVersion}</h2>

      {metricEntries.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 24 }}>
          <thead><tr>
            <th style={{ textAlign: 'left', padding: 8 }}>Metric</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Value</th>
          </tr></thead>
          <tbody>
            {metricEntries.map(([key, val]) => (
              <tr key={key} style={{ borderBottom: '1px solid #333' }}>
                <td style={{ padding: 8 }}>{key}</td>
                <td style={{ padding: 8, fontFamily: 'monospace' }}>
                  {typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Raw Data</h3>
      <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
