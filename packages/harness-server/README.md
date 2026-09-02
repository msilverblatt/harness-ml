# harness-server

MCP interface for Harness 2. It exposes 17 tools and 5 resources for project setup, data preparation, typed experiments, analysis, and version navigation.

```bash
harness-server
```

The server discovers a workspace by walking upward for `harness.yaml`; clients can also call `workspace.open` or `project.init`.
