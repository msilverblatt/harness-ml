import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { readThemeColors, type ThemeColors } from '../styles/colors';

export type ThemeName =
    | 'neon-dark' | 'neon-light'
    | 'brutalist-dark' | 'brutalist-light'
    | 'matrix' | 'claude'
    | 'openai' | 'solarized-dark' | 'catppuccin' | 'nord'
    | 'rosepine' | 'github-light';

export const THEME_LIST: { id: ThemeName; label: string; description: string }[] = [
    { id: 'neon-dark', label: 'Neon Dark', description: 'Blue neon on dark steel' },
    { id: 'neon-light', label: 'Neon Light', description: 'Clean light with blue accents' },
    { id: 'brutalist-dark', label: 'Brutalist Dark', description: 'Raw, no curves, high contrast' },
    { id: 'brutalist-light', label: 'Brutalist Light', description: 'Stark white, black borders' },
    { id: 'matrix', label: 'The Matrix', description: 'Green phosphor terminal' },
    { id: 'claude', label: 'Claude', description: 'Warm clay and orange' },
    { id: 'openai', label: 'ChatGPT', description: 'Clean dark with cool blue accents' },
    { id: 'solarized-dark', label: 'Solarized Dark', description: 'Ethan Schoonover\'s classic palette' },
    { id: 'catppuccin', label: 'Catppuccin', description: 'Pastel mocha on deep purple' },
    { id: 'nord', label: 'Nord', description: 'Arctic blue-grey tones' },
    { id: 'rosepine', label: 'Rosé Pine', description: 'Muted rose and gold on navy' },
    { id: 'github-light', label: 'GitHub Light', description: 'Clean white with blue links' },
];

const STORAGE_KEY = 'harness-studio-theme';
const DEFAULT_THEME: ThemeName = 'neon-dark';

interface ThemeContextValue {
    theme: ThemeName;
    setTheme: (t: ThemeName) => void;
    colors: ThemeColors;
}

export const ThemeContext = createContext<ThemeContextValue>({
    theme: DEFAULT_THEME,
    setTheme: () => {},
    colors: {} as ThemeColors,
});

export function useThemeProvider() {
    const [theme, setThemeState] = useState<ThemeName>(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        return (stored as ThemeName) || DEFAULT_THEME;
    });

    const [colors, setColors] = useState<ThemeColors>(() => readThemeColors());

    const setTheme = useCallback((t: ThemeName) => {
        localStorage.setItem(STORAGE_KEY, t);
        document.documentElement.setAttribute('data-theme', t);
        setThemeState(t);
    }, []);

    // Apply theme on mount and re-read colors when theme changes
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        // Small delay to let CSS cascade before reading computed values
        const frame = requestAnimationFrame(() => {
            setColors(readThemeColors());
        });
        return () => cancelAnimationFrame(frame);
    }, [theme]);

    return { theme, setTheme, colors };
}

export function useTheme() {
    return useContext(ThemeContext);
}
