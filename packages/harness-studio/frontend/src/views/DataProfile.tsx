import { useEffect, useState } from 'react';
import { api } from '../api';

export default function DataProfile({ selectedVersion: _sv, onSelectVersion: _osv }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [schema, setSchema] = useState<any>(null);
  const [profile, setProfile] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.data.schema().then(setSchema).catch(e => setError(e.message));
    api.data.profile().then(setProfile).catch(() => {});
  }, []);

  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!schema) return <div>Loading...</div>;

  const columns = schema.columns || schema.fields || (Array.isArray(schema) ? schema : []);

  return (
    <div>
      <h2>Data Schema</h2>

      {columns.length > 0 ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 24 }}>
          <thead><tr>
            <th style={{ textAlign: 'left', padding: 8 }}>Column</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Nullable</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Details</th>
          </tr></thead>
          <tbody>
            {columns.map((col: any, i: number) => (
              <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                <td style={{ padding: 8 }}>{col.name || col.column || `col-${i}`}</td>
                <td style={{ padding: 8 }}>{col.type || col.dtype || '-'}</td>
                <td style={{ padding: 8 }}>{col.nullable != null ? String(col.nullable) : '-'}</td>
                <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{JSON.stringify(col, null, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
          {JSON.stringify(schema, null, 2)}
        </pre>
      )}

      {profile && (
        <>
          <h2>Data Profile</h2>
          {typeof profile === 'object' && !Array.isArray(profile) ? (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={{ textAlign: 'left', padding: 8 }}>Stat</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Value</th>
              </tr></thead>
              <tbody>
                {Object.entries(profile).map(([key, val]) => (
                  <tr key={key} style={{ borderBottom: '1px solid #333' }}>
                    <td style={{ padding: 8 }}>{key}</td>
                    <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>
                      {typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
              {JSON.stringify(profile, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  );
}
