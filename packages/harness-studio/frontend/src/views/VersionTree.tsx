import { useEffect, useState } from 'react';
import { api } from '../api';

export default function VersionTree({ onSelectVersion }: { onSelectVersion?: (id: string) => void; selectedVersion?: string }) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.versions.tree().then(setData).catch(e => setError(e.message)); }, []);
  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!data) return <div>Loading...</div>;
  return (
    <div>
      <h2>Version Tree</h2>
      <p>Current: <strong>{data.current || 'none'}</strong></p>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead><tr>
          <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
          <th style={{ textAlign: 'left', padding: 8 }}>Parent</th>
          <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
          <th style={{ textAlign: 'left', padding: 8 }}>Hypothesis</th>
          <th style={{ textAlign: 'left', padding: 8 }}>Verdict</th>
          <th style={{ textAlign: 'left', padding: 8 }}>Metrics</th>
        </tr></thead>
        <tbody>
          {data.versions.map((v: any) => (
            <tr key={v.id} onClick={() => onSelectVersion?.(v.id)} style={{ cursor: 'pointer', borderBottom: '1px solid #333' }}>
              <td style={{ padding: 8, color: v.id === data.current ? '#4fc3f7' : '#ccc' }}>{v.id}</td>
              <td style={{ padding: 8 }}>{v.parent || '-'}</td>
              <td style={{ padding: 8 }}>{v.experiment_type || '-'}</td>
              <td style={{ padding: 8, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>{v.hypothesis || '-'}</td>
              <td style={{ padding: 8, color: v.verdict === 'improved' ? '#4caf50' : v.verdict === 'degraded' ? '#f44336' : '#888' }}>{v.verdict || '-'}</td>
              <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{JSON.stringify(v.metrics || {})}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
