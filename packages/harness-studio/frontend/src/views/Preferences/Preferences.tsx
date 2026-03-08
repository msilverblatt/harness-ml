import { useTheme, THEME_LIST, type ThemeName } from '../../hooks/useTheme';
import styles from './Preferences.module.css';

const THEME_SWATCHES: Record<ThemeName, { bg: string; accent: string; text: string }> = {
    'neon-dark': { bg: '#0d1117', accent: '#58a6ff', text: '#e6edf3' },
    'neon-light': { bg: '#f6f8fa', accent: '#0969da', text: '#1f2328' },
    'brutalist-dark': { bg: '#000000', accent: '#ffffff', text: '#ffffff' },
    'brutalist-light': { bg: '#ffffff', accent: '#000000', text: '#000000' },
    'matrix': { bg: '#000a00', accent: '#00ff41', text: '#00ff41' },
    'claude': { bg: '#faf6f1', accent: '#da7b28', text: '#2d2518' },
    'openai': { bg: '#171717', accent: '#ececec', text: '#ececec' },
    'solarized-dark': { bg: '#002b36', accent: '#268bd2', text: '#fdf6e3' },
    'catppuccin': { bg: '#1e1e2e', accent: '#cba6f7', text: '#cdd6f4' },
    'nord': { bg: '#2e3440', accent: '#88c0d0', text: '#eceff4' },
    'rosepine': { bg: '#191724', accent: '#ebbcba', text: '#e0def4' },
    'github-light': { bg: '#ffffff', accent: '#0969da', text: '#1f2328' },
};

export function PreferencesView() {
    const { theme, setTheme } = useTheme();

    return (
        <div className={styles.preferences}>
            <div className={styles.section}>
                <h2 className={styles.sectionTitle}>Theme</h2>
                <div className={styles.themeGrid}>
                    {THEME_LIST.map(t => {
                        const swatch = THEME_SWATCHES[t.id];
                        const active = theme === t.id;
                        return (
                            <button
                                key={t.id}
                                className={`${styles.themeCard} ${active ? styles.themeCardActive : ''}`}
                                onClick={() => setTheme(t.id)}
                            >
                                <div
                                    className={styles.themePreview}
                                    style={{ background: swatch.bg }}
                                >
                                    <div className={styles.previewHeader} style={{ background: swatch.accent, opacity: 0.15 }}>
                                        <div className={styles.previewDot} style={{ background: swatch.accent }} />
                                        <div className={styles.previewBar} style={{ background: swatch.accent, opacity: 0.4 }} />
                                    </div>
                                    <div className={styles.previewBody}>
                                        <div className={styles.previewLine} style={{ background: swatch.text, opacity: 0.6, width: '70%' }} />
                                        <div className={styles.previewLine} style={{ background: swatch.text, opacity: 0.3, width: '50%' }} />
                                        <div className={styles.previewLine} style={{ background: swatch.accent, width: '40%' }} />
                                    </div>
                                </div>
                                <div className={styles.themeInfo}>
                                    <span className={styles.themeName}>{t.label}</span>
                                    <span className={styles.themeDesc}>{t.description}</span>
                                </div>
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
