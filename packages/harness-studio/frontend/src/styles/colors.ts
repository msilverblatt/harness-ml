/**
 * Color constants for use in inline styles (recharts, ReactFlow, etc.)
 * Reads from CSS custom properties so themes are respected.
 */

const CSS_VAR_MAP = {
    bgPrimary: '--color-bg-primary',
    bgSecondary: '--color-bg-secondary',
    bgTertiary: '--color-bg-tertiary',
    bgHover: '--color-bg-hover',
    border: '--color-border',

    textPrimary: '--color-text-primary',
    textSecondary: '--color-text-secondary',
    textMuted: '--color-text-muted',

    accent: '--color-accent',
    success: '--color-success',
    warning: '--color-warning',
    error: '--color-error',
    orange: '--color-orange',

    nodeSource: '--color-node-source',
    nodeFeatures: '--color-node-features',
    nodeModel: '--color-node-model',
    nodeEnsemble: '--color-node-ensemble',
    nodeCalibration: '--color-node-calibration',
    nodeOutput: '--color-node-output',
} as const;

export type ThemeColors = Record<keyof typeof CSS_VAR_MAP, string>;

export function readThemeColors(): ThemeColors {
    const style = getComputedStyle(document.documentElement);
    const result = {} as Record<string, string>;
    for (const [key, cssVar] of Object.entries(CSS_VAR_MAP)) {
        result[key] = style.getPropertyValue(cssVar).trim();
    }
    return result as ThemeColors;
}
