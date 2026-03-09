import { useEffect } from 'react';

const SHORTCUTS: Record<string, string> = {
    '1': 'dashboard',
    '2': 'activity',
    '3': 'dag',
    '4': 'experiments',
    '5': 'diagnostics',
    '6': 'predictions',
    'r': 'refresh',
    '?': 'help',
};

export const SHORTCUT_DESCRIPTIONS: { key: string; label: string }[] = [
    { key: '1', label: 'Dashboard' },
    { key: '2', label: 'Activity' },
    { key: '3', label: 'DAG' },
    { key: '4', label: 'Experiments' },
    { key: '5', label: 'Diagnostics' },
    { key: '6', label: 'Predictions' },
    { key: 'R', label: 'Refresh current tab' },
    { key: '?', label: 'Show shortcuts' },
];

export function useKeyboardShortcuts(handlers: Record<string, () => void>) {
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Ignore when typing in inputs
            if (
                e.target instanceof HTMLInputElement ||
                e.target instanceof HTMLTextAreaElement ||
                e.target instanceof HTMLSelectElement
            ) {
                return;
            }
            // Ignore modifier combos (let browser handle Ctrl+R, etc.)
            if (e.ctrlKey || e.metaKey || e.altKey) return;

            const action = SHORTCUTS[e.key];
            if (action && handlers[action]) {
                e.preventDefault();
                handlers[action]();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [handlers]);
}
