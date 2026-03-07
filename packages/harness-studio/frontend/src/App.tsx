import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './styles/tokens.css';
import './styles/reset.css';
import { Layout } from './components/Layout/Layout';
import { Activity } from './views/Activity/Activity';

function Placeholder({ name }: { name: string }) {
    return <div style={{ padding: 'var(--space-4)', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>{name}</div>;
}

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route element={<Layout />}>
                    <Route index element={<Activity />} />
                    <Route path="dag" element={<Placeholder name="DAG" />} />
                    <Route path="experiments" element={<Placeholder name="Experiments" />} />
                    <Route path="diagnostics" element={<Placeholder name="Diagnostics" />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

export default App;
