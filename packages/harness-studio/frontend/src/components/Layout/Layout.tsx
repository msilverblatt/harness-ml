import { NavLink, Outlet } from 'react-router-dom';
import styles from './Layout.module.css';

const TABS = [
    { path: '/', label: 'Activity' },
    { path: '/dag', label: 'DAG' },
    { path: '/experiments', label: 'Experiments' },
    { path: '/diagnostics', label: 'Diagnostics' },
];

export function Layout() {
    return (
        <div className={styles.layout}>
            <header className={styles.header}>
                <span className={styles.logo}>Harness Studio</span>
                <nav className={styles.tabs}>
                    {TABS.map(t => (
                        <NavLink key={t.path} to={t.path} end
                            className={({ isActive }) =>
                                isActive ? styles.tabActive : styles.tab
                            }>
                            {t.label}
                        </NavLink>
                    ))}
                </nav>
            </header>
            <main className={styles.content}>
                <Outlet />
            </main>
        </div>
    );
}
