import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './styles/tokens.css';
import './styles/themes/neon-light.css';
import './styles/themes/brutalist-dark.css';
import './styles/themes/brutalist-light.css';
import './styles/themes/matrix.css';
import './styles/themes/claude.css';
import './styles/themes/openai.css';
import './styles/themes/solarized-dark.css';
import './styles/themes/catppuccin.css';
import './styles/themes/nord.css';
import './styles/themes/rosepine.css';
import './styles/themes/github-light.css';
import './styles/reset.css';
import { ThemeContext, useThemeProvider } from './hooks/useTheme';
import { Layout } from './components/Layout/Layout';
import { Dashboard } from './views/Dashboard/Dashboard';
import { Activity } from './views/Activity/Activity';
import { DAG } from './views/DAG/DAG';
import { Experiments } from './views/Experiments/Experiments';
import { Diagnostics } from './views/Diagnostics/Diagnostics';
import { DataView } from './views/Data/Data';
import { FeaturesView } from './views/Features/Features';
import { ModelsView } from './views/Models/Models';
import { EnsembleView } from './views/Ensemble/Ensemble';
import { PredictionsView } from './views/Predictions/Predictions';
import { ConfigView } from './views/Config/Config';
import { PreferencesView } from './views/Preferences/Preferences';
import { ProjectRedirect } from './components/ProjectRedirect';
import { ToastProvider } from './components/Toast/Toast';

function App() {
    const themeCtx = useThemeProvider();

    return (
        <ThemeContext.Provider value={themeCtx}>
            <ToastProvider>
            <BrowserRouter>
                <Routes>
                    {/* Redirect root to most recent project */}
                    <Route index element={<ProjectRedirect />} />
                    {/* Project-scoped routes */}
                    <Route path=":project" element={<Layout />}>
                        <Route index element={<Navigate to="dashboard" replace />} />
                        <Route path="dashboard" element={<Dashboard />} />
                        <Route path="activity" element={<Activity />} />
                        <Route path="dag" element={<DAG />} />
                        <Route path="experiments" element={<Experiments />} />
                        <Route path="data" element={<DataView />} />
                        <Route path="features" element={<FeaturesView />} />
                        <Route path="models" element={<ModelsView />} />
                        <Route path="ensemble" element={<EnsembleView />} />
                        <Route path="predictions" element={<PredictionsView />} />
                        <Route path="diagnostics" element={<Diagnostics />} />
                        <Route path="config" element={<ConfigView />} />
                        <Route path="preferences" element={<PreferencesView />} />
                    </Route>
                </Routes>
            </BrowserRouter>
            </ToastProvider>
        </ThemeContext.Provider>
    );
}

export default App;
