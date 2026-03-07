import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './styles/tokens.css';
import './styles/reset.css';
import { Layout } from './components/Layout/Layout';
import { Dashboard } from './views/Dashboard/Dashboard';
import { Activity } from './views/Activity/Activity';
import { DAG } from './views/DAG/DAG';
import { Experiments } from './views/Experiments/Experiments';
import { Diagnostics } from './views/Diagnostics/Diagnostics';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route element={<Layout />}>
                    <Route index element={<Dashboard />} />
                    <Route path="activity" element={<Activity />} />
                    <Route path="dag" element={<DAG />} />
                    <Route path="experiments" element={<Experiments />} />
                    <Route path="diagnostics" element={<Diagnostics />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

export default App;
