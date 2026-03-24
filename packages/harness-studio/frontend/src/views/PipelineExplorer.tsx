import { useEffect, useState } from 'react';
import { api } from '../api';

export default function PipelineExplorer({ selectedVersion: _sv, onSelectVersion: _osv }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [config, setConfig] = useState<any>(null);
  const [dag, setDag] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.pipeline.config().then(setConfig).catch(e => setError(e.message));
    api.pipeline.dag().then(setDag).catch(() => {});
  }, []);

  if (error) return <div style={{ color: '#f44336' }}>Error: {error}</div>;
  if (!config) return <div>Loading...</div>;

  return (
    <div>
      <h2>Pipeline Configuration</h2>

      {config.models && (
        <section style={{ marginBottom: 24 }}>
          <h3>Models</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={{ textAlign: 'left', padding: 8 }}>Name</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Parameters</th>
            </tr></thead>
            <tbody>
              {(Array.isArray(config.models) ? config.models : [config.models]).map((m: any, i: number) => (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: 8 }}>{m.name || m.type || `model-${i}`}</td>
                  <td style={{ padding: 8 }}>{m.type || '-'}</td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{JSON.stringify(m.params || m.parameters || m, null, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {config.features && (
        <section style={{ marginBottom: 24 }}>
          <h3>Features</h3>
          <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
            {JSON.stringify(config.features, null, 2)}
          </pre>
        </section>
      )}

      {config.ensemble && (
        <section style={{ marginBottom: 24 }}>
          <h3>Ensemble</h3>
          <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
            {JSON.stringify(config.ensemble, null, 2)}
          </pre>
        </section>
      )}

      {dag && (
        <section style={{ marginBottom: 24 }}>
          <h3>DAG</h3>
          <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
            {JSON.stringify(dag, null, 2)}
          </pre>
        </section>
      )}

      <section>
        <h3>Full Config</h3>
        <pre style={{ background: '#111', padding: 12, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
          {JSON.stringify(config, null, 2)}
        </pre>
      </section>
    </div>
  );
}
