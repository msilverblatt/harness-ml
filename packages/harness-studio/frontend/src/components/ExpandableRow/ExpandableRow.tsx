import { useState } from 'react';
import styles from './ExpandableRow.module.css';

interface ExpandableRowProps {
    children: React.ReactNode;
    detail: React.ReactNode;
    defaultOpen?: boolean;
}

export function ExpandableRow({ children, detail, defaultOpen = false }: ExpandableRowProps) {
    const [open, setOpen] = useState(defaultOpen);

    return (
        <div className={styles.row}>
            <div className={styles.summary} onClick={() => setOpen(prev => !prev)}>
                <span className={`${styles.chevron} ${open ? styles.chevronOpen : ''}`}>
                    &#9656;
                </span>
                {children}
            </div>
            {open && (
                <div className={styles.detail}>
                    {detail}
                </div>
            )}
        </div>
    );
}
