import { useEffect, useState } from 'react';
import { api } from '../api';

export default function MCPMonitor({ selectedVersion: _sv, onSelectVersion: _osv }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [events, setEvents] = useState<any[] | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.monitor.events().then(setEvents).catch(e => setError(e.message));
    api.monitor.stats().then(setStats).catch(() => {});
  }, []);

  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!events) return <div>Loading...</div>;

  return (
    <div>
      <h2>MCP Monitor</h2>

      {stats && (
        <section style={{ marginBottom: 24 }}>
          <h3>Stats</h3>
          <table style={{ borderCollapse: 'collapse' }}>
            <tbody>
              {Object.entries(stats).map(([key, val]) => (
                <tr key={key} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: 8, fontWeight: 'bold' }}>{key}</td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>
                    {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <h3>Events ({events.length})</h3>
      {events.length === 0 ? (
        <div style={{ color: '#888' }}>No events recorded.</div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr>
            <th style={{ textAlign: 'left', padding: 8 }}>Time</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Tool</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Status</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Details</th>
          </tr></thead>
          <tbody>
            {events.map((ev: any, i: number) => (
              <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                <td style={{ padding: 8, whiteSpace: 'nowrap', fontSize: 12 }}>{ev.timestamp || ev.time || '-'}</td>
                <td style={{ padding: 8 }}>{ev.type || ev.event_type || '-'}</td>
                <td style={{ padding: 8 }}>{ev.tool || ev.tool_name || '-'}</td>
                <td style={{ padding: 8, color: ev.status === 'error' ? '#f44336' : ev.status === 'success' ? '#4caf50' : '#ccc' }}>
                  {ev.status || '-'}
                </td>
                <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12, maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {typeof ev.data === 'object' ? JSON.stringify(ev.data) : String(ev.data ?? ev.message ?? '-')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
