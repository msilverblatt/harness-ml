import { useLayoutContext } from '../../components/Layout/Layout';
import { StatBar } from './StatBar';
import { EventLog } from './EventLog';
import styles from './Activity.module.css';

export function Activity() {
    const { events, connected } = useLayoutContext();

    return (
        <div className={styles.activity}>
            <StatBar connected={connected} />
            <EventLog wsEvents={events} />
        </div>
    );
}
