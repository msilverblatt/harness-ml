import type { ReactNode } from 'react';
import styles from './nodeStyles.module.css';

interface NodeShellProps {
    typeClass: string;
    data: Record<string, unknown>;
    children: ReactNode;
    style?: React.CSSProperties;
}

export function NodeShell({ typeClass, data, children, style }: NodeShellProps) {
    const isRunning = !!data._running;
    const isSelected = !!data._selected;

    const className = [
        styles.node,
        typeClass,
        isRunning && styles.nodeRunning,
        isSelected && styles.nodeSelected,
    ].filter(Boolean).join(' ');

    return (
        <div className={className} style={style}>
            {children}
        </div>
    );
}
