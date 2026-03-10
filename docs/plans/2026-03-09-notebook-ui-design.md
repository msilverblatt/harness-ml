# Notebook Studio UI — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement the plan created from this design.

**Goal:** Add a Notebook tab to Studio and integrate notebook content into Dashboard and Activity tabs so the agent's narrative (theory, plan, findings) is the primary lens for understanding what's happening.

**Architecture:** New Notebook page with two views (timeline + journal), redesigned Dashboard hero with narrative context, and enhanced Activity tab with context bar + inline notebook entries. All backed by existing `GET /notebook` REST endpoint + WebSocket refresh.

---

## 1. Notebook Tab (Agent sidebar section)

### Two Views (toggle at top)

**Timeline View:**
- Reverse-chronological feed of all entries
- Each entry: type badge (color-coded), relative timestamp (full datetime on hover), markdown content, auto-tag pills, experiment ID link
- Struck entries hidden by default; inline indicator ("N previous versions") that expands to show them with strikethrough + dimmed styling + struck reason

**Journal View:**
- Latest theory + latest plan pinned as hero cards at top (visually distinct)
- Below: findings, decisions, and research grouped by experiment ID
- Entries without experiment_id in a "General" section
- Same struck entry behavior as timeline

### Structured Metadata (extracted from body, rendered as UI elements)

- **Type badge:** Color-coded pill
  - theory = purple
  - finding = blue
  - plan = green
  - decision = orange
  - research = teal
  - note = gray
- **Auto-tags:** Clickable pills — clicking filters to that tag
- **Experiment ID:** Clickable link navigating to the Experiments tab
- **Timestamp:** Relative time ("2 hours ago") with full datetime on hover

### Filtering (persistent across both views)

- Type filter (multi-select pills)
- Free text search
- Tag filter (dropdown/pill selector from all known auto-tags)
- Date range picker
- Experiment ID filter

---

## 2. Dashboard (redesigned hierarchy)

**Narrative leads, data supports.**

**Top section (full width):**
- Current theory card
- Current plan card
- Last 2-3 findings

**Below:** Existing metrics widgets, mini DAG, activity summary unchanged.

---

## 3. Activity Tab (enhanced)

**Context bar (pinned at top of Activity tab):**
- Current theory (latest)
- Current plan (latest)
- Running/latest experiment: ID, hypothesis, status

**Inline in activity stream:**
- When agent writes a notebook entry, it appears as a visually distinct activity item (not a raw tool call)
- Shows type badge, content preview, auto-tags
- Weaves narrative into the action log so you see *when* the agent updated its thinking

---

## 4. Data Flow

- **REST:** `GET /notebook` returns all entries including struck
- **WebSocket refresh:** Re-fetch notebook data when `notebook` tool completes
- **Dashboard/Activity context:** Derive latest theory, plan, and experiment from notebook + experiments endpoints
- **Cross-tab links:** Experiment ID links in notebook entries navigate to Experiments tab

---

## 5. Styling

- CSS Modules (consistent with all other Studio views)
- Design tokens from `tokens.css`
- Dark mode default (inherits theme system)
- Type badge colors as new CSS custom properties
