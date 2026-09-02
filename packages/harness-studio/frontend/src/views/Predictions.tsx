import { useEffect, useState } from 'react';
import { api } from '../api';

const PAGE_SIZE = 50;

export default function Predictions({ selectedVersion }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [data, setData] = useState<any>(null);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedVersion) return;
    setData(null);
    setError(null);
    setOffset(0);
    api.predictions(selectedVersion, PAGE_SIZE, 0).then(setData).catch(e => setError(e.message));
  }, [selectedVersion]);

  useEffect(() => {
    if (!selectedVersion || offset === 0) return;
    api.predictions(selectedVersion, PAGE_SIZE, offset).then(setData).catch(e => setError(e.message));
  }, [selectedVersion, offset]);

  if (!selectedVersion) return <div style={{ color: '#888' }}>Select a version from the Version Tree tab to view predictions.</div>;
  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!data) return <div>Loading...</div>;

  const rows = data.predictions || data.rows || (Array.isArray(data) ? data : []);
  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  return (
    <div>
      <h2>Predictions: {selectedVersion}</h2>
      <div style={{ marginBottom: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
        <button onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} disabled={offset === 0}
          style={{ padding: '4px 12px', background: '#333', color: '#ccc', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          Prev
        </button>
        <span>Offset: {offset}</span>
        <button onClick={() => setOffset(offset + PAGE_SIZE)} disabled={rows.length < PAGE_SIZE}
          style={{ padding: '4px 12px', background: '#333', color: '#ccc', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          Next
        </button>
        {data.total != null && <span style={{ color: '#888' }}>Total: {data.total}</span>}
      </div>

      {rows.length === 0 ? (
        <div style={{ color: '#888' }}>No predictions found.</div>
      ) : (
        <div style={{ overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              {columns.map(c => <th key={c} style={{ textAlign: 'left', padding: 8 }}>{c}</th>)}
            </tr></thead>
            <tbody>
              {rows.map((row: any, i: number) => (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  {columns.map(c => (
                    <td key={c} style={{ padding: 8, fontFamily: 'monospace', fontSize: 12, whiteSpace: 'nowrap' }}>
                      {typeof row[c] === 'object' ? JSON.stringify(row[c]) : String(row[c] ?? '-')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
