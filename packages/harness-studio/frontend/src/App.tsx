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
import { Notebook } from './views/Notebook/Notebook';
import { ConfigView } from './views/Config/Config';
import { PreferencesView } from './views/Preferences/Preferences';
import { ProjectRedirect } from './components/ProjectRedirect';
import { ToastProvider } from './components/Toast/Toast';
import { ErrorBoundary } from './components/ErrorBoundary/ErrorBoundary';

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
                        <Route path="dashboard" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
                        <Route path="activity" element={<ErrorBoundary><Activity /></ErrorBoundary>} />
                        <Route path="dag" element={<ErrorBoundary><DAG /></ErrorBoundary>} />
                        <Route path="experiments" element={<ErrorBoundary><Experiments /></ErrorBoundary>} />
                        <Route path="notebook" element={<ErrorBoundary><Notebook /></ErrorBoundary>} />
                        <Route path="data" element={<ErrorBoundary><DataView /></ErrorBoundary>} />
                        <Route path="features" element={<ErrorBoundary><FeaturesView /></ErrorBoundary>} />
                        <Route path="models" element={<ErrorBoundary><ModelsView /></ErrorBoundary>} />
                        <Route path="ensemble" element={<ErrorBoundary><EnsembleView /></ErrorBoundary>} />
                        <Route path="predictions" element={<ErrorBoundary><PredictionsView /></ErrorBoundary>} />
                        <Route path="diagnostics" element={<ErrorBoundary><Diagnostics /></ErrorBoundary>} />
                        <Route path="config" element={<ErrorBoundary><ConfigView /></ErrorBoundary>} />
                        <Route path="preferences" element={<ErrorBoundary><PreferencesView /></ErrorBoundary>} />
                    </Route>
                </Routes>
            </BrowserRouter>
            </ToastProvider>
        </ThemeContext.Provider>
    );
}

export default App;
