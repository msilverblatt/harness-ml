import { useState } from 'react';

const THEMES = [
  { name: 'Dark', bg: '#0a0a0a', fg: '#e0e0e0' },
  { name: 'Light', bg: '#fafafa', fg: '#222' },
  { name: 'Blue', bg: '#0d1b2a', fg: '#c8d6e5' },
];

export default function Preferences({ selectedVersion: _sv, onSelectVersion: _osv }: { selectedVersion?: string; onSelectVersion?: (id: string) => void }) {
  const [theme, setTheme] = useState(THEMES[0].name);

  return (
    <div>
      <h2>Preferences</h2>

      <section style={{ marginTop: 16 }}>
        <h3>Theme</h3>
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          {THEMES.map(t => (
            <button key={t.name} onClick={() => setTheme(t.name)}
              style={{
                padding: '8px 16px',
                border: theme === t.name ? '2px solid #4fc3f7' : '2px solid #333',
                borderRadius: 4,
                cursor: 'pointer',
                background: t.bg,
                color: t.fg,
              }}>
              {t.name}
            </button>
          ))}
        </div>
        <p style={{ marginTop: 8, color: '#888', fontSize: 14 }}>
          Active theme: <strong>{theme}</strong> (theme application coming in a future update)
        </p>
      </section>
    </div>
  );
}
