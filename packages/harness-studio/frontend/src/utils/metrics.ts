const LOWER_IS_BETTER = new Set(['brier', 'brier_score', 'ece', 'log_loss', 'mae', 'mse', 'rmse']);

export function isLowerBetter(name: string): boolean {
    return LOWER_IS_BETTER.has(name);
}

export function isImprovement(name: string, delta: number): boolean {
    if (delta === 0) return false;
    return LOWER_IS_BETTER.has(name) ? delta < 0 : delta > 0;
}
