import { useState, useEffect, useRef } from 'react';
import styles from './ExpandableRow.module.css';

interface ExpandableRowProps {
    children: React.ReactNode;
    detail: React.ReactNode;
    defaultOpen?: boolean;
    autoExpand?: boolean;
}

export function ExpandableRow({ children, detail, defaultOpen = false, autoExpand }: ExpandableRowProps) {
    const [open, setOpen] = useState(defaultOpen || autoExpand === true);
    const userToggled = useRef(false);

    useEffect(() => {
        if (autoExpand === undefined) return;
        if (autoExpand) {
            setOpen(true);
            userToggled.current = false;
        } else if (!userToggled.current) {
            setOpen(false);
        }
    }, [autoExpand]);

    const handleToggle = () => {
        userToggled.current = true;
        setOpen(prev => !prev);
    };

    return (
        <div className={styles.row}>
            <div className={styles.summary} onClick={handleToggle}>
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
