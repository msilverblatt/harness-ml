# HarnessML Claude Code Plugin Design

## What

A Claude Code plugin (`packages/harness-plugin/`) that exposes harnessml-runner's
config_writer functions as MCP tools. Enables AI-driven ML experimentation
where the user describes what they want and Claude handles all pipeline
mechanics automatically.

## Architecture

```
packages/harness-plugin/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── .mcp.json                    # MCP server definition (stdio)
├── src/harnessml/plugin/
│   ├── __init__.py
│   └── mcp_server.py            # FastMCP server wrapping config_writer
├── skills/
│   └── ml-workflow/
│       └── SKILL.md             # ML experimentation workflow skill
└── pyproject.toml               # uv workspace package, depends on harnessml-runner
```

## MCP Server

`mcp_server.py` uses FastMCP to register ~20 tools that map 1:1 to
`harnessml.runner.config_writer` functions. Each tool:

- Accepts `project_dir` (defaults to cwd if not specified)
- Calls the corresponding config_writer function
- Returns a markdown string (already built into config_writer)

### Tools

| Tool | config_writer function | Category |
|------|----------------------|----------|
| `add_model` | `config_writer.add_model()` | Models |
| `remove_model` | `config_writer.remove_model()` | Models |
| `show_models` | `config_writer.show_models()` | Models |
| `show_presets` | `config_writer.show_presets()` | Models |
| `configure_ensemble` | `config_writer.configure_ensemble()` | Ensemble |
| `configure_backtest` | `config_writer.configure_backtest()` | Backtest |
| `add_dataset` | `config_writer.add_dataset()` | Data |
| `profile_data` | `config_writer.profile_data()` | Data |
| `available_features` | `config_writer.available_features()` | Data |
| `add_feature` | `config_writer.add_feature()` | Features |
| `add_features_batch` | `config_writer.add_features_batch()` | Features |
| `test_transformations` | `config_writer.test_feature_transformations()` | Features |
| `discover_features` | `config_writer.discover_features()` | Features |
| `experiment_create` | `config_writer.experiment_create()` | Experiments |
| `write_overlay` | `config_writer.write_overlay()` | Experiments |
| `show_config` | `config_writer.show_config()` | Config |
| `list_runs` | `config_writer.list_runs()` | Runs |

### project_dir Resolution

1. If `project_dir` param is provided, use it
2. Otherwise, use cwd
3. Validate that `config/` directory exists before proceeding

## Skill: ml-workflow

Teaches Claude the ML experimentation pattern:

1. **Scaffold** → `scaffold_project` (one-time setup)
2. **Ingest** → `add_dataset` (bring in new data)
3. **Discover** → `discover_features`, `available_features` (explore what's useful)
4. **Transform** → `test_transformations`, `add_feature` (engineer features)
5. **Model** → `add_model` with preset, `configure_ensemble` (build models)
6. **Experiment** → `experiment_create`, `write_overlay` (test hypotheses)
7. **Evaluate** → `show_config`, `list_runs` (review results)

## Dependencies

- `harnessml-runner` (workspace dependency, provides config_writer)
- `mcp` (already in workspace venv, provides FastMCP)

## Plugin Installation

After building, install via:
```bash
claude plugin install /path/to/packages/harness-plugin
```

Or add to `.claude/settings.json` plugins list.
