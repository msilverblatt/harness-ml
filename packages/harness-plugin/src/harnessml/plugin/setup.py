"""harness-setup — zero-friction first-time experience."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

import click

_BANNER = r"""
 ██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗    ███╗   ███╗██╗
 ██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝    ████╗ ████║██║
 ███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗    ██╔████╔██║██║
 ██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██║
 ██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║    ██║ ╚═╝ ██║███████╗
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝
"""

_DEMO_PROMPT = (
    "I have a California housing dataset at harness-demo/data/raw/housing.csv. "
    "Initialize a regression project to predict median_house_value. "
    "Explore the data, build a diverse set of models, and run a backtest."
)

_SKILLS = [
    "harness-run-experiment",
    "harness-explore-space",
    "harness-domain-research",
]


def _find_repo_root() -> Path | None:
    """Walk up from CWD looking for the harness-ml repo root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        root = Path(result.stdout.strip())
        if (root / "packages" / "harness-plugin").exists():
            return root
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def _find_package_root() -> Path:
    """Find the harness-plugin package root (for pip installs)."""
    return Path(__file__).parent


def _write_mcp_config(repo_root: Path | None) -> Path:
    """Write .mcp.json to CWD."""
    mcp_path = Path.cwd() / ".mcp.json"
    existing = {}
    if mcp_path.exists():
        try:
            existing = json.loads(mcp_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    servers = existing.get("mcpServers", {})

    if repo_root:
        servers["harness-ml"] = {
            "command": "uv",
            "args": ["run", "--directory", str(repo_root), "harness-ml"],
        }
    else:
        servers["harness-ml"] = {
            "command": "harness-ml",
            "args": [],
        }

    existing["mcpServers"] = servers
    mcp_path.write_text(json.dumps(existing, indent=2) + "\n")
    return mcp_path


def _install_skills(repo_root: Path | None) -> int:
    """Copy skill files into .claude/skills/ for auto-discovery."""
    skills_target = Path.cwd() / ".claude" / "skills"
    installed = 0

    # Find skill source files
    if repo_root:
        skills_source = repo_root / "skills"
    else:
        skills_source = _find_package_root() / "skills"

    if not skills_source.exists():
        return 0

    for skill_name in _SKILLS:
        source_file = skills_source / skill_name / "SKILL.md"
        if not source_file.exists():
            continue
        target_dir = skills_target / skill_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_dir / "SKILL.md")
        installed += 1

    return installed


def _create_demo_project() -> Path:
    """Create harness-demo/ with the bundled CSV."""
    demo_dir = Path.cwd() / "harness-demo"
    data_dir = demo_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_target = data_dir / "housing.csv"
    if not csv_target.exists():
        csv_source = _find_package_root() / "demo_data" / "housing.csv"
        shutil.copy2(csv_source, csv_target)

    return demo_dir


def _start_studio(project_dir: Path, port: int) -> subprocess.Popen | None:
    """Start harness-studio in the background."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "harnessml.studio.cli",
             "--project-dir", str(project_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        return None


def _has_claude() -> bool:
    """Check if claude CLI is available."""
    return shutil.which("claude") is not None


@click.command()
@click.option("--port", default=8421, help="Studio port")
@click.option("--no-studio", is_flag=True, help="Skip starting Studio")
@click.option("--no-claude", is_flag=True, help="Skip launching Claude Code")
def main(port: int, no_studio: bool, no_claude: bool):
    """Set up HarnessML for first use — MCP config, skills, demo project, and Studio."""
    click.echo(_BANNER)

    repo_root = _find_repo_root()

    # 1. MCP config
    mcp_path = _write_mcp_config(repo_root)
    click.echo(f"  \u2713 MCP config written to {mcp_path.name}")

    # 2. Skills
    n_skills = _install_skills(repo_root)
    if n_skills:
        click.echo(f"  \u2713 Skills installed ({n_skills} skills)")

    # 3. Demo project
    demo_dir = _create_demo_project()
    click.echo(f"  \u2713 Demo project created at ./{demo_dir.name}/")

    # 4. Studio
    studio_proc = None
    if not no_studio:
        studio_proc = _start_studio(demo_dir, port)
        if studio_proc:
            click.echo(f"  \u2713 Studio started at http://localhost:{port}")
            import time
            time.sleep(1)
            webbrowser.open(f"http://localhost:{port}")
        else:
            click.echo("  \u26a0 Could not start Studio (harness-studio not found)")

    click.echo()

    # 5. Launch Claude
    if no_claude or not _has_claude():
        if not _has_claude() and not no_claude:
            click.echo("  Claude Code not found. Install it, then run:")
            click.echo()
        click.echo(f'  claude "{_DEMO_PROMPT}"')
        click.echo()
        return

    click.echo("  Press Enter to launch Claude Code with this prompt:")
    click.echo()
    click.echo(f'    "{_DEMO_PROMPT}"')
    click.echo()
    input("  [Enter] ")

    os.execvp("claude", ["claude", _DEMO_PROMPT])


if __name__ == "__main__":
    main()
