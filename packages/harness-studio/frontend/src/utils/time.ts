/**
 * Humanize an ISO timestamp to a relative time string.
 * Returns "just now", "Xm ago", "Xh ago", "Xd ago", or a formatted date.
 */
export function humanizeTimestamp(iso: string): string {
    const date = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
}

/**
 * Format an ISO timestamp to a full locale-aware date/time string.
 * Suitable for tooltips showing the exact time.
 */
export function formatTimestamp(iso: string): string {
    return new Date(iso).toLocaleString();
}
