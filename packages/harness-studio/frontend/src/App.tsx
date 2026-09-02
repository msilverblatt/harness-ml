import { useState } from 'react';
import VersionTree from './views/VersionTree';
import VersionDetail from './views/VersionDetail';
import PipelineExplorer from './views/PipelineExplorer';
import Diagnostics from './views/Diagnostics';
import Predictions from './views/Predictions';
import DataProfile from './views/DataProfile';
import MCPMonitor from './views/MCPMonitor';
import Preferences from './views/Preferences';

const VIEWS = [
  { name: 'Version Tree', Component: VersionTree },
  { name: 'Version Detail', Component: VersionDetail },
  { name: 'Pipeline', Component: PipelineExplorer },
  { name: 'Diagnostics', Component: Diagnostics },
  { name: 'Predictions', Component: Predictions },
  { name: 'Data', Component: DataProfile },
  { name: 'MCP Monitor', Component: MCPMonitor },
  { name: 'Preferences', Component: Preferences },
];

export default function App() {
  const [active, setActive] = useState(0);
  const [selectedVersion, setSelectedVersion] = useState<string | undefined>(undefined);
  const View = VIEWS[active].Component;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{ display: 'flex', gap: 2, padding: '8px 16px', background: '#1a1a1a', borderBottom: '1px solid #333' }}>
        <span style={{ fontWeight: 'bold', marginRight: 24, color: '#fff' }}>Harness Studio</span>
        {VIEWS.map((v, i) => (
          <button key={i} onClick={() => setActive(i)}
            style={{
              padding: '6px 12px', border: 'none', borderRadius: 4, cursor: 'pointer',
              background: active === i ? '#333' : 'transparent', color: active === i ? '#fff' : '#888',
            }}>
            {v.name}
          </button>
        ))}
      </header>
      <main style={{ flex: 1, padding: 16, overflow: 'auto' }}>
        <View selectedVersion={selectedVersion} onSelectVersion={setSelectedVersion} />
      </main>
    </div>
  );
}
