/**
 * TypeScript types for all Studio API responses.
 *
 * These types mirror the backend response shapes from the FastAPI routes.
 * They are used throughout the frontend to provide type safety.
 */

// --- Events ---

export interface Event {
    id: number;
    timestamp: string;
    tool: string;
    action: string;
    params: Record<string, unknown>;
    result: string;
    duration_ms: number;
    status: 'running' | 'success' | 'error' | 'progress';
    project?: string;
    caller?: string;
}

export interface EventStats {
    total_calls: number;
    errors: number;
    by_tool: Record<string, number>;
}

// --- Project ---

export interface ProjectStatus {
    project_name: string;
    task: string;
    target_column: string | null;
    model_types_tried: number;
    active_models: number;
    experiments_run: number;
    run_count: number;
    feature_count: number;
    latest_metrics: Record<string, number>;
}

export interface ProjectInfo {
    name: string;
    project_dir: string;
    last_seen: number;
}

// --- Runs ---

export interface RunSummary {
    id: string;
    metrics: Record<string, number>;
    metric_std?: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    has_report?: boolean;
    experiment_id?: string | null;
}

export interface RunMetrics {
    metrics: Record<string, number>;
    meta_coefficients?: Record<string, number>;
    error?: string;
}

export interface FoldBreakdownResponse {
    folds: Record<string, unknown>[];
    metric_names: string[];
    error?: string;
}

export interface CorrelationResponse {
    models: string[];
    matrix: number[][];
    error?: string;
}

export interface CalibrationPoint {
    bin_center: number;
    predicted: number;
    actual: number;
    count: number;
}

export interface CalibrationResponse {
    calibration: CalibrationPoint[];
    prob_column: string;
    error?: string;
}

export interface ResidualPoint {
    predicted: number;
    actual: number;
    residual: number;
}

export interface ResidualHistogramBin {
    bin_center: number;
    count: number;
}

export interface ResidualResponse {
    scatter: ResidualPoint[];
    histogram: ResidualHistogramBin[];
    stats: {
        mean_residual: number;
        std_residual: number;
        median_residual: number;
        max_overpredict: number;
        max_underpredict: number;
    };
    error?: string;
}

// --- Experiments ---

export interface Experiment {
    experiment_id: string;
    timestamp?: string;
    description?: string;
    hypothesis?: string;
    conclusion?: string;
    verdict?: string;
    metrics?: Record<string, number>;
    metric_std?: Record<string, number>;
    baseline_metrics?: Record<string, number>;
    primary_delta?: number;
    primary_metric?: string;
    notes?: string;
}

export interface ExperimentDetail {
    id: string;
    hypothesis?: string;
    conclusion?: string;
    overlay?: Record<string, unknown>;
    results?: Record<string, unknown>;
}

// --- Notebook ---

export type NotebookEntryType = 'theory' | 'finding' | 'research' | 'decision' | 'plan' | 'note';

export interface NotebookEntry {
    id: string;
    type: NotebookEntryType;
    timestamp: string;
    content: string;
    tags: string[];
    auto_tags: string[];
    struck: boolean;
    struck_reason: string | null;
    struck_at: string | null;
    experiment_id: string | null;
}

// --- Predictions ---

export interface PredictionSummary {
    run_id: string;
    total_predictions: number;
    model_columns: string[];
    correct?: number;
    accuracy?: number;
    avg_confidence?: number;
    fold_counts?: Record<string, number>;
}

export interface PredictionDistribution {
    run_id: string;
    prob_column: string;
    histogram: { bin_center: number; count: number }[];
    stats: { mean: number; std: number; median: number };
    error?: string;
}

export interface PredictionPage {
    run_id: string;
    columns: string[];
    rows: Record<string, unknown>[];
    total: number;
    page: number;
    page_size: number;
}

// --- DAG ---

export interface DagNode {
    id: string;
    type: string;
    label: string;
    data: Record<string, unknown>;
}

export interface DagEdge {
    source: string;
    target: string;
}

export interface DagResponse {
    nodes: DagNode[];
    edges: DagEdge[];
}

// --- Models ---

export interface ModelConfig {
    name: string;
    type: string;
    mode?: string;
    params: Record<string, unknown>;
    features: string[];
    feature_count: number;
    active: boolean;
    include_in_ensemble: boolean;
    source: 'production' | 'experimental';
}

// --- Features ---

export interface Feature {
    name: string;
    type: string;
    source?: string;
    derived_from: string[];
    formula?: string | null;
    category: string;
    enabled: boolean;
    used_by?: string[];
}

// --- Ensemble ---

export interface EnsembleConfig {
    method: string;
    meta_learner: Record<string, unknown>;
    temperature?: number;
    clip_floor?: number;
    calibration: Record<string, unknown>;
    cv_strategy?: string;
    fold_column?: string;
    metrics: string[];
    exclude_models: string[];
    pre_calibration: Record<string, unknown>;
    task_type?: string;
    target_column?: string;
    model_count: number;
}

export interface EnsembleWeights {
    run_id: string;
    coefficients: Record<string, number>;
    pooled_metrics: Record<string, unknown>;
    task_type?: string;
    error?: string;
}

// --- Picks ---

export interface PickAnalysisResponse {
    total: number;
    correct: number;
    accuracy: number;
    avg_confidence: number;
    avg_agreement?: number;
    avg_confidence_correct?: number;
    avg_confidence_incorrect?: number;
    error?: string;
}

// --- Error response ---

export interface ApiError {
    error: string;
    category?: 'transient' | 'not_found' | 'permanent' | 'server';
    retryable?: boolean;
}
