import styles from './StatBox.module.css';

interface StatBoxProps {
    label: string;
    value: string | number;
    subtitle?: string;
    variant?: 'default' | 'success' | 'warning' | 'error';
}

export function StatBox({ label, value, subtitle, variant = 'default' }: StatBoxProps) {
    const valueClass = variant !== 'default'
        ? `${styles.value} ${styles[variant]}`
        : styles.value;

    return (
        <div className={styles.box}>
            <div className={styles.label}>{label}</div>
            <div className={valueClass}>{value}</div>
            {subtitle && <div className={styles.subtitle}>{subtitle}</div>}
        </div>
    );
}
