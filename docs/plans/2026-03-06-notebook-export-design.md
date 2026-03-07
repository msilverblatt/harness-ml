# Notebook Export & Cloud Upload Design

## Summary

Add notebook generation and cloud upload capabilities to harnessml, enabling users
to export their pipeline as a Jupyter notebook and optionally upload data/notebooks
to Google Drive or Kaggle.

## Components

### 1. Notebook Builder (`core/runner/notebook.py`)

Generates a `.ipynb` from a project config. The notebook mirrors the harnessml
pipeline (not standalone code) — it imports harness-core and runs the same
stages.

**Destination parameter** controls platform-specific cells:

| Destination | Data loading | Install cell | Output paths |
|-------------|-------------|--------------|--------------|
| `colab` | `drive.mount()` + Drive paths | `!pip install harness-core` | `/content/drive/...` |
| `kaggle` | `/kaggle/input/<dataset>/` | `!pip install -q harness-core` | `/kaggle/working/` |
| `local` | Local filesystem paths | None (assumes installed) | Project directory |

**Notebook structure (cells):**

1. Platform setup (install deps, mount drive / set paths)
2. Imports
3. Load config (inline YAML or from file)
4. Load data
5. Feature engineering
6. Train models
7. Evaluate (metrics, calibration)
8. Results summary

### 2. Google Drive Adapter (`core/runner/drives/drive.py`)

Handles Google Drive operations via `google-api-python-client`.

**Capabilities:**
- OAuth browser flow with token caching (per-project)
- Upload file(s) to Drive (any file — data, notebooks, configs)
- Create folders
- Get shareable link / Colab URL for `.ipynb` files
- List files in a folder

**Dependencies (optional):**
- `google-api-python-client`
- `google-auth-oauthlib`

Graceful error if not installed.

### 3. Kaggle Adapter (`core/runner/drives/kaggle.py`)

Handles Kaggle operations via `kaggle` CLI/API.

**Capabilities:**
- Upload dataset (create or update)
- Upload notebook (kernel push)
- Auth via `~/.kaggle/kaggle.json` (standard Kaggle auth)

**Dependencies (optional):**
- `kaggle` package

### 4. MCP Integration

**New actions on existing tools:**

| Tool | Action | Description |
|------|--------|-------------|
| `pipeline` | `export_notebook` | Generate `.ipynb` from project config |
| `data` | `upload_drive` | Upload file(s) to Google Drive |
| `data` | `upload_kaggle` | Upload file(s) / dataset to Kaggle |

**`pipeline(action="export_notebook")` params:**
- `destination`: `colab` | `kaggle` | `local` (required)
- `output_path`: where to save the `.ipynb` locally (optional, defaults to project dir)

**`data(action="upload_drive")` params:**
- `files`: list of file paths to upload
- `folder_id`: Drive folder ID (optional)
- `folder_name`: create/find folder by name (optional)

**`data(action="upload_kaggle")` params:**
- `files`: list of file paths to upload
- `dataset_slug`: Kaggle dataset identifier
- `title`: dataset title

## Auth Flows

### Google Drive
1. First call triggers OAuth browser flow
2. Token cached at `<project_dir>/.harnessml/drive_token.json`
3. Subsequent calls reuse cached token (auto-refresh)
4. User needs a Google Cloud project with Drive API enabled + OAuth client ID

### Kaggle
1. Standard `~/.kaggle/kaggle.json` with API key
2. No additional setup needed if user already has Kaggle CLI configured

## Data Strategy

When exporting a notebook:
- The notebook references data by path (Drive path, Kaggle input path, or local path)
- User uploads data separately via `data(action="upload_drive|upload_kaggle")`
- Notebook builder generates the correct loading code based on destination

This keeps notebook generation decoupled from upload — you can generate without
uploading, or upload without generating a notebook.

## File Layout

```
packages/harness-core/src/harnessml/core/runner/
  notebook.py          # Notebook generation
  drives/
    drive.py           # Google Drive adapter
    kaggle.py          # Kaggle adapter
```

## Out of Scope (Tier 2, future)

- Colab Enterprise programmatic execution (`gcloud colab executions create`)
- Kaggle kernel execution + result retrieval
- Automatic data upload as part of notebook export (keep decoupled)
