"""Multi-format export for competition results.

Supports markdown, JSON, and CSV output for brackets, standings,
probabilities, and analysis reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from easyml.sports.competitions.schemas import (
    CompetitionResult,
    CompetitionStructure,
    MatchupContext,
    StandingsEntry,
)


def export_bracket_markdown(
    result: CompetitionResult,
    entity_names: dict[str, str],
    structure: CompetitionStructure,
    output_dir: Path,
    title: str = "Competition Bracket",
) -> Path:
    """Export bracket picks as human-readable markdown.

    Organized by round. Each matchup shows:
    **{pick}** over {loser} ({prob}%, agreement: {agreement}%) [UPSET]

    Header includes strategy, expected_points, win_probability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bracket.md"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Strategy:** {result.strategy}")
    lines.append(f"**Expected Points:** {result.expected_points:.2f}")
    lines.append(f"**Win Probability:** {result.win_probability:.2%}")
    lines.append("")

    # Group matchups by round
    rounds: dict[int, list[tuple[str, Any]]] = {}
    for slot, matchup_data in result.matchups.items():
        if isinstance(matchup_data, MatchupContext):
            ctx = matchup_data
        elif isinstance(matchup_data, dict):
            ctx = MatchupContext(**matchup_data)
        else:
            continue
        round_num = ctx.round_num
        rounds.setdefault(round_num, []).append((slot, ctx))

    round_names = structure.config.rounds if structure.config.rounds else []

    for round_num in sorted(rounds.keys()):
        if round_num - 1 < len(round_names):
            round_label = round_names[round_num - 1]
        else:
            round_label = f"Round {round_num}"
        lines.append(f"## {round_label}")
        lines.append("")

        for slot, ctx in rounds[round_num]:
            pick_id = ctx.pick
            loser_id = ctx.entity_b if pick_id == ctx.entity_a else ctx.entity_a
            pick_name = entity_names.get(pick_id, pick_id)
            loser_name = entity_names.get(loser_id, loser_id)

            prob = ctx.prob_a if pick_id == ctx.entity_a else (1.0 - ctx.prob_a)
            agreement = ctx.model_agreement

            upset_marker = " [UPSET]" if ctx.upset else ""
            lines.append(
                f"- **{pick_name}** over {loser_name} "
                f"({prob:.0%}, agreement: {agreement:.0%}){upset_marker}"
            )

        lines.append("")

    path.write_text("\n".join(lines))
    return path


def export_standings_markdown(
    standings: list[StandingsEntry],
    entity_names: dict[str, str],
    output_dir: Path,
    title: str = "Competition Standings",
) -> Path:
    """Export standings as a markdown table with rank, entity, W, L, D, Pts, GD."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "standings.md"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| Rank | Entity | W | L | D | Pts | GD |")
    lines.append("|------|--------|---|---|---|-----|-----|")

    for rank, entry in enumerate(standings, start=1):
        name = entity_names.get(entry.entity, entry.entity)
        gd = f"{entry.goal_diff:+.0f}" if entry.goal_diff != 0 else "0"
        lines.append(
            f"| {rank} | {name} | {entry.wins} | {entry.losses} | "
            f"{entry.draws} | {entry.points:.0f} | {gd} |"
        )

    lines.append("")
    path.write_text("\n".join(lines))
    return path


def export_json(
    results: list[CompetitionResult] | list[StandingsEntry],
    entity_names: dict[str, str],
    output_dir: Path,
    filename: str = "results.json",
) -> Path:
    """Export results as structured JSON.

    For CompetitionResult: picks, picks_named, matchups with full context.
    For StandingsEntry: ranked list with entity names resolved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if not results:
        path.write_text(json.dumps([], indent=2))
        return path

    first = results[0]

    if isinstance(first, CompetitionResult):
        output = []
        for r in results:
            assert isinstance(r, CompetitionResult)
            picks_named = {
                slot: entity_names.get(eid, eid) for slot, eid in r.picks.items()
            }
            matchups_serialized: dict[str, Any] = {}
            for slot, m in r.matchups.items():
                if isinstance(m, MatchupContext):
                    matchups_serialized[slot] = m.model_dump()
                elif isinstance(m, dict):
                    matchups_serialized[slot] = m
                else:
                    matchups_serialized[slot] = str(m)

            output.append(
                {
                    "strategy": r.strategy,
                    "expected_points": r.expected_points,
                    "win_probability": r.win_probability,
                    "top10_probability": r.top10_probability,
                    "picks": r.picks,
                    "picks_named": picks_named,
                    "matchups": matchups_serialized,
                }
            )
        path.write_text(json.dumps(output, indent=2))
    elif isinstance(first, StandingsEntry):
        output_standings = []
        for rank, entry in enumerate(results, start=1):
            assert isinstance(entry, StandingsEntry)
            output_standings.append(
                {
                    "rank": rank,
                    "entity": entry.entity,
                    "entity_name": entity_names.get(entry.entity, entry.entity),
                    "wins": entry.wins,
                    "losses": entry.losses,
                    "draws": entry.draws,
                    "points": entry.points,
                    "goal_diff": entry.goal_diff,
                }
            )
        path.write_text(json.dumps(output_standings, indent=2))

    return path


def export_csv(
    probabilities: pd.DataFrame,
    output_dir: Path,
    filename: str = "probabilities.csv",
) -> Path:
    """Export probabilities DataFrame as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    probabilities.to_csv(path, index=False)
    return path


def export_analysis_report(
    result: CompetitionResult,
    pick_stories: list[dict],
    entity_profiles: list[dict],
    confidence_report: dict,
    entity_names: dict[str, str],
    output_dir: Path,
    title: str = "Competition Analysis",
) -> Path:
    """Comprehensive markdown combining confidence diagnostics, entity profiles,
    and bracket picks. Generic -- no sports-specific terminology.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis.md"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Strategy:** {result.strategy}")
    lines.append(f"**Expected Points:** {result.expected_points:.2f}")
    lines.append(f"**Win Probability:** {result.win_probability:.2%}")
    lines.append("")

    # --- Confidence Diagnostics ---
    lines.append("## Confidence Diagnostics")
    lines.append("")
    for key, value in confidence_report.items():
        if isinstance(value, list):
            lines.append(f"### {key}")
            lines.append("")
            for item in value:
                if isinstance(item, dict):
                    parts = [f"{k}: {v}" for k, v in item.items()]
                    lines.append(f"- {', '.join(parts)}")
                else:
                    lines.append(f"- {item}")
            lines.append("")
        else:
            lines.append(f"- **{key}:** {value}")
    lines.append("")

    # --- Entity Profiles ---
    lines.append("## Entity Profiles")
    lines.append("")
    for profile in entity_profiles:
        eid = profile.get("entity", "")
        name = entity_names.get(eid, eid)
        lines.append(f"### {name}")
        lines.append("")
        for k, v in profile.items():
            if k == "entity":
                continue
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # --- Pick Stories ---
    lines.append("## Pick Details")
    lines.append("")
    for story in pick_stories:
        slot = story.get("slot", "")
        narrative = story.get("narrative", "")
        lines.append(f"### {slot}")
        lines.append("")
        lines.append(narrative)
        lines.append("")
        for k, v in story.items():
            if k in ("slot", "narrative"):
                continue
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path
