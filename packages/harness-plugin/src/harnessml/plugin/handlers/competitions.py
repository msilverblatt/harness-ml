"""Handler for manage_competitions tool."""
from __future__ import annotations

import json

from harnessml.plugin.handlers._common import parse_json_param
from harnessml.plugin.handlers._validation import (
    validate_enum,
    validate_required,
    collect_hints,
    format_response_with_hints,
)


# ---------------------------------------------------------------------------
# In-memory registry of created competitions
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {}


def _get_competition(name: str) -> dict | None:
    """Look up a competition by name."""
    return _REGISTRY.get(name)


def _set_competition(name: str, data: dict) -> None:
    """Store a competition in the registry."""
    _REGISTRY[name] = data


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_create(*, config, name=None, **_kwargs):
    """Create a competition from a config dict."""
    err = validate_required(config, "config")
    if err:
        return err

    parsed = parse_json_param(config)
    if not isinstance(parsed, dict):
        return "**Error**: `config` must be a JSON object."

    from harnessml.sports.competitions.schemas import CompetitionConfig

    comp_name = name or parsed.get("name", "default")

    try:
        comp_config = CompetitionConfig(**{
            k: v for k, v in parsed.items() if k != "name"
        })
    except Exception as e:
        return f"**Error**: Invalid competition config: {e}"

    _set_competition(comp_name, {
        "name": comp_name,
        "config": comp_config,
        "structure": None,
        "simulator": None,
        "sim_results": None,
        "round_probs": None,
        "brackets": None,
    })

    fmt = comp_config.format.value
    n = comp_config.n_participants

    lines = [
        f"## Competition Created: `{comp_name}`",
        "",
        f"- **Format:** {fmt}",
        f"- **Participants:** {n}",
        f"- **Seeding:** {comp_config.seeding.value}",
    ]
    if comp_config.rounds:
        lines.append(f"- **Rounds:** {', '.join(comp_config.rounds)}")
    if comp_config.regions:
        lines.append(f"- **Regions:** {', '.join(comp_config.regions)}")
    if comp_config.scoring:
        lines.append(f"- **Scoring:** {comp_config.scoring.type}")

    return "\n".join(lines)


def _handle_list_formats(**_kwargs):
    """Show available competition formats."""
    from harnessml.sports.competitions.schemas import CompetitionFormat

    lines = [
        "## Available Competition Formats",
        "",
        "| Format | Description |",
        "|--------|-------------|",
    ]

    descriptions = {
        CompetitionFormat.single_elimination: "Standard knockout bracket; losers are eliminated immediately.",
        CompetitionFormat.double_elimination: "Two-loss elimination; losers drop to a losers bracket.",
        CompetitionFormat.round_robin: "Every participant plays every other participant.",
        CompetitionFormat.swiss: "Paired by record each round; fewer rounds than round-robin.",
        CompetitionFormat.group_knockout: "Group stage (round-robin) followed by knockout bracket.",
    }

    for fmt in CompetitionFormat:
        desc = descriptions.get(fmt, "")
        lines.append(f"| `{fmt.value}` | {desc} |")

    return "\n".join(lines)


def _handle_simulate(*, name=None, n_sims=None, seed=None, **_kwargs):
    """Run Monte Carlo simulations."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found. Create it first with action='create'."

    if comp.get("simulator") is None:
        return (
            f"**Error**: Competition `{comp_name}` has no simulator attached. "
            "Build the structure and attach probabilities before simulating."
        )

    simulator = comp["simulator"]
    n = n_sims or 10_000
    s = seed or 42

    results = simulator.simulate_many(n, seed=s)
    comp["sim_results"] = results

    # Compute round probs
    round_probs = simulator.entity_round_probabilities(n_sims=n, seed=s)
    comp["round_probs"] = round_probs

    # Summary stats
    final_round = max(simulator.structure.round_slots.keys())
    final_slots = simulator.structure.round_slots[final_round]

    champ_counts: dict[str, int] = {}
    for sim in results:
        for slot in final_slots:
            winner = sim.get(slot, "")
            champ_counts[winner] = champ_counts.get(winner, 0) + 1

    top_champs = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    lines = [
        f"## Simulation Complete: `{comp_name}`",
        "",
        f"- **Simulations:** {n:,}",
        f"- **Seed:** {s}",
        f"- **Unique champions:** {len(champ_counts)}",
        "",
        "### Top Championship Probabilities",
        "",
        "| Entity | Probability |",
        "|--------|-------------|",
    ]
    for entity, count in top_champs:
        prob = count / n
        lines.append(f"| {entity} | {prob:.2%} |")

    return "\n".join(lines)


def _handle_standings(*, name=None, top_n=None, **_kwargs):
    """Get standings distributions from simulation results."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    round_probs = comp.get("round_probs")
    if round_probs is None:
        return f"**Error**: No simulation results for `{comp_name}`. Run simulate first."

    n = top_n or 20
    if "champion" in round_probs.columns:
        sorted_df = round_probs.sort_values("champion", ascending=False).head(n)
    else:
        sorted_df = round_probs.head(n)

    cols = [c for c in round_probs.columns if c != "entity"]

    lines = [
        f"## Standings: `{comp_name}`",
        "",
    ]

    header = "| Entity | " + " | ".join(cols) + " |"
    sep = "|--------|" + "|".join(["----------" for _ in cols]) + "|"
    lines.append(header)
    lines.append(sep)

    for _, row in sorted_df.iterrows():
        entity = str(row["entity"])
        vals = " | ".join(f"{float(row[c]):.2%}" for c in cols)
        lines.append(f"| {entity} | {vals} |")

    return "\n".join(lines)


def _handle_round_probs(*, name=None, top_n=None, **_kwargs):
    """Entity progression probabilities per round."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    round_probs = comp.get("round_probs")
    if round_probs is None:
        return f"**Error**: No simulation results for `{comp_name}`. Run simulate first."

    n = top_n or 20
    if "champion" in round_probs.columns:
        sorted_df = round_probs.sort_values("champion", ascending=False).head(n)
    else:
        sorted_df = round_probs.head(n)

    cols = [c for c in round_probs.columns if c != "entity"]

    lines = [
        f"## Round Progression Probabilities: `{comp_name}`",
        "",
    ]

    header = "| Entity | " + " | ".join(cols) + " |"
    sep = "|--------|" + "|".join(["----------" for _ in cols]) + "|"
    lines.append(header)
    lines.append(sep)

    for _, row in sorted_df.iterrows():
        entity = str(row["entity"])
        vals = " | ".join(f"{float(row[c]):.2%}" for c in cols)
        lines.append(f"| {entity} | {vals} |")

    return "\n".join(lines)


def _handle_generate_brackets(
    *, name=None, pool_size=None, n_brackets=None, n_sims=None, seed=None, **_kwargs
):
    """Generate pool-aware brackets."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    if comp.get("simulator") is None:
        return f"**Error**: Competition `{comp_name}` has no simulator attached."

    err = validate_required(pool_size, "pool_size")
    if err:
        return err

    simulator = comp["simulator"]
    config = comp["config"]

    if config.scoring is None:
        return "**Error**: Competition has no scoring config. Set scoring in the config."

    from harnessml.sports.competitions.optimizer import CompetitionOptimizer

    optimizer = CompetitionOptimizer(simulator, config.scoring)
    brackets = optimizer.generate_brackets(
        pool_size=pool_size,
        n_brackets=n_brackets or 3,
        n_sims=n_sims or 10_000,
        seed=seed or 42,
    )

    comp["brackets"] = brackets

    lines = [
        f"## Generated Brackets: `{comp_name}`",
        "",
        f"- **Pool size:** {pool_size}",
        f"- **Brackets generated:** {len(brackets)}",
        "",
    ]

    for i, br in enumerate(brackets, 1):
        lines.append(f"### Bracket {i}: {br.strategy}")
        lines.append(f"- Expected points: {br.expected_points:.2f}")
        lines.append(f"- Win probability: {br.win_probability:.2%}")
        lines.append(f"- Top 10% probability: {br.top10_probability:.2%}")
        lines.append("")

    return "\n".join(lines)


def _handle_score_bracket(*, name=None, picks=None, actuals=None, **_kwargs):
    """Score bracket picks against actuals."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    err = validate_required(picks, "picks")
    if err:
        return err
    err = validate_required(actuals, "actuals")
    if err:
        return err

    parsed_picks = parse_json_param(picks)
    parsed_actuals = parse_json_param(actuals)

    if not isinstance(parsed_picks, dict) or not isinstance(parsed_actuals, dict):
        return "**Error**: `picks` and `actuals` must be JSON objects mapping slot -> winner."

    config = comp["config"]
    structure = comp.get("structure")

    if structure is None:
        return f"**Error**: Competition `{comp_name}` has no structure. Build it first."
    if config.scoring is None:
        return "**Error**: Competition has no scoring config."

    from harnessml.sports.competitions.scorer import CompetitionScorer

    scorer = CompetitionScorer(config.scoring)
    result = scorer.score_bracket(parsed_picks, parsed_actuals, structure)

    lines = [
        f"## Score: `{comp_name}`",
        "",
        f"- **Total Points:** {result.total_points:.1f}",
        "",
        "### Per-Round Breakdown",
        "",
        "| Round | Correct | Total | Points |",
        "|-------|---------|-------|--------|",
    ]

    for rk in sorted(result.round_total.keys()):
        correct = result.round_correct.get(rk, 0)
        total = result.round_total.get(rk, 0)
        pts = result.round_points.get(rk, 0.0)
        lines.append(f"| {rk} | {correct} | {total} | {pts:.1f} |")

    return "\n".join(lines)


def _handle_adjust(*, name=None, adjustments=None, **_kwargs):
    """Apply probability adjustments."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    err = validate_required(adjustments, "adjustments")
    if err:
        return err

    parsed = parse_json_param(adjustments)
    if not isinstance(parsed, dict):
        return "**Error**: `adjustments` must be a JSON object."

    return (
        f"## Adjustments Registered: `{comp_name}`\n\n"
        f"- **Entity multipliers:** {len(parsed.get('entity_multipliers', {}))}\n"
        f"- **Probability overrides:** {len(parsed.get('probability_overrides', {}))}\n"
        f"- **External weight:** {parsed.get('external_weight', 0.0)}"
    )


def _handle_explain(*, name=None, **_kwargs):
    """Generate pick explanations."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    brackets = comp.get("brackets")
    if not brackets:
        return f"**Error**: No brackets generated for `{comp_name}`. Run generate_brackets first."

    bracket = brackets[0]
    n_matchups = len(bracket.matchups)

    lines = [
        f"## Pick Explanations: `{comp_name}`",
        "",
        f"- **Strategy:** {bracket.strategy}",
        f"- **Matchups explained:** {n_matchups}",
        "",
    ]

    for slot, matchup_data in bracket.matchups.items():
        from harnessml.sports.competitions.schemas import MatchupContext
        if isinstance(matchup_data, MatchupContext):
            ctx = matchup_data
        elif isinstance(matchup_data, dict):
            ctx = MatchupContext(**matchup_data)
        else:
            continue

        opponent = ctx.entity_b if ctx.pick == ctx.entity_a else ctx.entity_a
        prob = ctx.prob_a if ctx.pick == ctx.entity_a else 1.0 - ctx.prob_a
        upset_tag = " **[UPSET]**" if ctx.upset else ""

        lines.append(f"### {slot}")
        lines.append(
            f"**{ctx.pick}** over {opponent} ({prob:.0%}, "
            f"agreement: {ctx.model_agreement:.0%}){upset_tag}"
        )
        lines.append("")

    return "\n".join(lines)


def _handle_profiles(*, name=None, top_n=None, **_kwargs):
    """Entity profiles from simulation."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    round_probs = comp.get("round_probs")
    if round_probs is None:
        return f"**Error**: No simulation results for `{comp_name}`. Run simulate first."

    n = top_n or 20
    if "champion" not in round_probs.columns:
        return f"**Error**: Round probabilities missing 'champion' column."

    sorted_df = round_probs.sort_values("champion", ascending=False).head(n)
    structure = comp.get("structure")

    lines = [
        f"## Entity Profiles: `{comp_name}` (top {n})",
        "",
    ]

    for _, row in sorted_df.iterrows():
        entity = str(row["entity"])
        seed = ""
        if structure:
            seed = structure.entity_to_seed.get(entity, "")
        champ_prob = float(row["champion"])
        seed_str = f" ({seed})" if seed else ""
        lines.append(f"### {entity}{seed_str}")
        lines.append(f"- Champion probability: {champ_prob:.2%}")

        round_cols = [c for c in round_probs.columns if c not in ("entity", "champion")]
        for col in round_cols:
            lines.append(f"- {col}: {float(row[col]):.2%}")
        lines.append("")

    return "\n".join(lines)


def _handle_confidence(*, name=None, **_kwargs):
    """Pre-competition diagnostics."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    if comp.get("simulator") is None:
        return f"**Error**: Competition `{comp_name}` has no simulator attached."

    from harnessml.sports.competitions.confidence import compute_model_disagreement

    simulator = comp["simulator"]
    disagreements = compute_model_disagreement(simulator)

    lines = [
        f"## Confidence Diagnostics: `{comp_name}`",
        "",
        f"### Model Disagreement (top {len(disagreements)} matchups)",
        "",
        "| Slot | Entity A | Entity B | Agreement | Ensemble Prob |",
        "|------|----------|----------|-----------|---------------|",
    ]

    for d in disagreements:
        lines.append(
            f"| {d['slot']} | {d['entity_a']} | {d['entity_b']} | "
            f"{d['agreement']:.2%} | {d['prob_ensemble']:.2%} |"
        )

    return "\n".join(lines)


def _handle_export(*, name=None, output_dir=None, format_type=None, **_kwargs):
    """Export competition results."""
    comp_name = name or "default"
    comp = _get_competition(comp_name)
    if comp is None:
        return f"**Error**: Competition `{comp_name}` not found."

    err = validate_required(output_dir, "output_dir")
    if err:
        return err

    from pathlib import Path
    out = Path(output_dir)
    fmt = format_type or "json"

    brackets = comp.get("brackets")
    if not brackets:
        return f"**Error**: No brackets to export for `{comp_name}`. Generate them first."

    from harnessml.sports.competitions.export import export_json

    entity_names: dict[str, str] = {}
    structure = comp.get("structure")
    if structure:
        entity_names = {v: v for v in structure.seed_to_entity.values()}

    if fmt == "json":
        path = export_json(brackets, entity_names, out)
        return f"Exported to `{path}`"
    elif fmt == "markdown":
        from harnessml.sports.competitions.export import export_bracket_markdown
        if structure is None:
            return "**Error**: No structure available for markdown export."
        path = export_bracket_markdown(brackets[0], entity_names, structure, out)
        return f"Exported to `{path}`"
    elif fmt == "csv":
        round_probs = comp.get("round_probs")
        if round_probs is None:
            return "**Error**: No round probabilities to export as CSV."
        from harnessml.sports.competitions.export import export_csv
        path = export_csv(round_probs, out)
        return f"Exported to `{path}`"
    else:
        return f"**Error**: Unknown format `{fmt}`. Use: json, markdown, csv."


def _handle_list_strategies(**_kwargs):
    """Show available bracket generation strategies."""
    from harnessml.sports.competitions.optimizer import BUILTIN_STRATEGIES

    descriptions = {
        "chalk": "Always pick the favorite in every matchup.",
        "near_chalk": "Chalk with random flips for close matchups (underdog >40%).",
        "random_sim": "Single stochastic simulation using model probabilities.",
        "contrarian": "Compress probabilities toward 0.5 (uniform upset boost).",
        "late_contrarian": "Ramp upset boost by round (more upsets in later rounds).",
        "champion_anchor": "Force a specific entity to win all its games.",
    }

    lines = [
        "## Available Strategies",
        "",
        "### Built-in",
        "",
        "| Strategy | Description |",
        "|----------|-------------|",
    ]

    for name in sorted(BUILTIN_STRATEGIES.keys()):
        desc = descriptions.get(name, "")
        lines.append(f"| `{name}` | {desc} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

ACTIONS = {
    "create": _handle_create,
    "list_formats": _handle_list_formats,
    "simulate": _handle_simulate,
    "standings": _handle_standings,
    "round_probs": _handle_round_probs,
    "generate_brackets": _handle_generate_brackets,
    "score_bracket": _handle_score_bracket,
    "adjust": _handle_adjust,
    "explain": _handle_explain,
    "profiles": _handle_profiles,
    "confidence": _handle_confidence,
    "export": _handle_export,
    "list_strategies": _handle_list_strategies,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_competitions action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="competitions", **kwargs)
    return format_response_with_hints(result, hints)
