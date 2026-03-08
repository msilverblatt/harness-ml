import { useLayoutContext } from '../../components/Layout/Layout';
import { StatBar } from './StatBar';
import { EventLog } from './EventLog';
import styles from './Activity.module.css';

export function Activity() {
    const { events } = useLayoutContext();

    return (
        <div className={styles.activity}>
            <StatBar />
            <EventLog wsEvents={events} />
        </div>
    );
}
