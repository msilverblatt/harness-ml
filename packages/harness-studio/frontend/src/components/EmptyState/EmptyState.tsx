import styles from './EmptyState.module.css';

interface EmptyStateAction {
    label: string;
    onClick: () => void;
}

interface EmptyStateProps {
    icon?: string;
    title: string;
    description: string;
    action?: EmptyStateAction;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
    return (
        <div className={styles.emptyState}>
            {icon && <span className={styles.icon}>{icon}</span>}
            <h3 className={styles.title}>{title}</h3>
            <p className={styles.description}>{description}</p>
            {action && (
                <button className={styles.action} onClick={action.onClick}>
                    {action.label}
                </button>
            )}
        </div>
    );
}
