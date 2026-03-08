import { useState, useRef, useCallback, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { lookupGlossary, formatMetricLabel } from './glossary';
import styles from './Tooltip.module.css';

interface TooltipProps {
    /** The glossary key to look up (e.g. "brier", "max_depth"). */
    term: string;
    /**
     * Optional override for the visible label text.
     * If omitted, the raw `term` is displayed.
     */
    label?: ReactNode;
}

interface PopupPos {
    top: number;
    left: number;
}

/**
 * Inline tooltip for ML / statistics terms.
 *
 * Renders the term with a dotted underline. On hover a portal-based
 * popover shows the glossary definition above the term — never clipped
 * by overflow:hidden ancestors.
 */
export function Tooltip({ term, label }: TooltipProps) {
    const definition = lookupGlossary(term);
    const spanRef = useRef<HTMLSpanElement>(null);
    const [pos, setPos] = useState<PopupPos | null>(null);

    const formatted = formatMetricLabel(term);
    const displayLabel = label ?? (
        typeof formatted === 'object' ? formatted.text : formatted
    );

    const show = useCallback(() => {
        const el = spanRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        setPos({
            top: rect.top + window.scrollY,
            left: rect.left + rect.width / 2,
        });
    }, []);

    const hide = useCallback(() => setPos(null), []);

    if (!definition) {
        return <>{displayLabel}</>;
    }

    return (
        <>
            <span
                ref={spanRef}
                className={styles.wrapper}
                onMouseEnter={show}
                onMouseLeave={hide}
            >
                <span className={styles.term}>{displayLabel}</span>
            </span>
            {pos && createPortal(
                <div
                    className={styles.popup}
                    style={{ top: pos.top, left: pos.left }}
                >
                    <div className={styles.popupTitle}>
                        {typeof displayLabel === 'string' ? displayLabel : term}
                    </div>
                    <div className={styles.popupBody}>{definition}</div>
                </div>,
                document.body,
            )}
        </>
    );
}

/**
 * Helper to render a metric name with automatic R² formatting and
 * glossary tooltip.
 */
export function MetricLabel({ name }: { name: string }) {
    const formatted = formatMetricLabel(name);
    const display = typeof formatted === 'object' ? formatted.text : formatted;
    return <Tooltip term={name} label={display} />;
}
