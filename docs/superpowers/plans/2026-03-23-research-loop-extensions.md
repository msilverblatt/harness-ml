# research-loop Extensions Plan (Package 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend research-loop to support harness 2's version-tree experiment model: parent version selection, dynamic per-experiment baselines, conclude-as-terminal (no promote/discard), and new verdict vocabulary.

**Architecture:** Modifications to the existing research-loop TypeScript codebase. Changes touch types, state manager, schema merging, loop workflow, prompts, and tests.

**Tech Stack:** TypeScript, Zod, protomcp, Vitest

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Section 13 (research-loop Extensions)

**Codebase:** `/Users/msilverblatt/Projects/harness2/packages/research-loop/`

---

## Changes Required

### 1. Verdict Vocabulary
**Current:** `['keep', 'discard', 'inconclusive']`
**New:** `['improved', 'degraded', 'inconclusive', 'mixed']`
**Files:** `types.ts`, `schema.ts`, `prompts.ts`, tests

### 2. Parent Version Selection
**Current:** Single global baseline, no concept of "parent"
**New:** `propose` accepts optional `parent` field (opaque — harness passes version ID, research-loop stores it)
**Files:** `types.ts` (ExperimentState gets `parent` field), `schema.ts` (propose schema), `loop.ts` (pass parent to hooks), `state.ts`

### 3. Dynamic Baseline
**Current:** `getBaseline()` called once at propose time, returns global baseline
**New:** `getBaseline(parentId?)` receives the parent ID so harness can return the parent version's metrics
**Files:** `types.ts` (hook signature), `loop.ts` (pass parent to getBaseline)

### 4. Conclude as Terminal
**Current:** `conclude → [promote | discard]` (promote/discard are terminal)
**New:** `conclude` IS terminal. No promote/discard steps.
**Files:** `loop.ts` (workflow step definitions), `state.ts` (finalize on conclude), `prompts.ts`, tests

---

### Task 1: Verdict Vocabulary + Parent Field

**Files to modify:**
- `src/types.ts`
- `src/schema.ts`
- `src/prompts.ts`
- `tests/types.test.ts`
- `tests/schema.test.ts`
- `tests/prompts.test.ts`

- [ ] **Step 1: Update types.ts**

Change VERDICTS and add parent to ExperimentState:
```typescript
// Change from:
export const VERDICTS = ['keep', 'discard', 'inconclusive'] as const;
// To:
export const VERDICTS = ['improved', 'degraded', 'inconclusive', 'mixed'] as const;

// Add to ExperimentState:
interface ExperimentState {
  // ... existing fields ...
  parent?: string;  // Parent version/experiment ID (opaque, passed by domain)
}

// Add to ExperimentSummary:
interface ExperimentSummary {
  // ... existing fields ...
  parent?: string;
}
```

- [ ] **Step 2: Update schema.ts**

Add `parent` to propose schema merge:
```typescript
function mergeProposeSchema(domainSchema?) {
  // Add parent as optional string alongside hypothesis and description
  const baseFields = {
    hypothesis: z.string(),
    description: z.string().optional(),
    parent: z.string().optional(),  // NEW
  };
  // ... rest stays same but verdict enum uses new values
}

function mergeConcludeSchema(domainSchema?) {
  // Update verdict enum to new values
  const baseFields = {
    conclusion: z.string(),
    verdict: z.enum(['improved', 'degraded', 'inconclusive', 'mixed']),
  };
}
```

Update `PLUGIN_OWNED_FIELDS` to include `'parent'`.

- [ ] **Step 3: Update prompts.ts**

Update `how-to-conclude` prompt to explain new verdicts:
```
- improved: The experiment made things measurably better
- degraded: The experiment made things worse
- inconclusive: Results are ambiguous or not statistically significant
- mixed: Some dimensions improved, others degraded
```

- [ ] **Step 4: Update tests**

Fix all tests that reference old verdicts (`keep`, `discard`) to use new ones (`improved`, `degraded`).
Add test for `parent` field in propose schema.

- [ ] **Step 5: Run tests**

```bash
cd packages/research-loop && npm test
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(research-loop): new verdict vocabulary + parent field in propose"
```

---

### Task 2: Dynamic Baseline + Conclude as Terminal

**Files to modify:**
- `src/types.ts`
- `src/state.ts`
- `src/loop.ts`
- `tests/state.test.ts`
- `tests/loop.test.ts`
- `tests/e2e.test.ts`

- [ ] **Step 1: Update getBaseline hook signature**

In `types.ts`:
```typescript
interface ResearchLoopHooks {
  // Change from:
  getBaseline?: () => unknown;
  // To:
  getBaseline?: (parentId?: string) => unknown;

  // Remove promote/discard hooks (no longer needed):
  // onPromote is removed
  // onDiscard is removed

  // Add optional onConclude hook:
  onConclude?: (experiment: ExperimentSummary, ctx: ExperimentContext) => void;
}
```

- [ ] **Step 2: Update state.ts**

Add `parent` to create():
```typescript
create(hypothesis: string, proposeArgs?: Record<string, unknown>, parent?: string): ExperimentState {
  // ... existing logic ...
  // Store parent on the experiment
  experiment.parent = parent;
}
```

- [ ] **Step 3: Update loop.ts — restructure workflow**

The workflow changes from 5 steps to 3 steps:
```
propose (initial) → setup_and_run → conclude (terminal)
```

Remove `promote` and `discard` steps entirely.

In `propose` handler:
```typescript
// Extract parent from args
const parent = args.parent as string | undefined;
// Pass parent to getBaseline
if (hooks.getBaseline) {
  const baseline = hooks.getBaseline(parent);
  state.setBaseline(baseline);
}
// Pass parent to state.create()
state.create(hypothesis, domainArgs, parent);
```

In `conclude` handler (NOW TERMINAL):
```typescript
// conclude is now terminal — it finalizes the experiment
state.conclude(verdict, conclusion);
const summary = state.finalize();  // Move to history immediately
if (hooks.onConclude) {
  hooks.onConclude(summary, ctx);
}
if (hooks.onLog) {
  hooks.onLog(summary, ctx);
}
```

- [ ] **Step 4: Update tests**

- state.test.ts: test parent field stored and retrieved
- loop.test.ts: workflow has 3 steps (not 5), conclude is terminal, no promote/discard
- e2e.test.ts: full flow is propose → run → conclude (terminal)

- [ ] **Step 5: Run tests**

```bash
cd packages/research-loop && npm test
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(research-loop): dynamic baseline + conclude as terminal (3-step workflow)"
```

---

### Task 3: Update Examples + E2E Verification

- [ ] **Step 1: Update iris example to use new API**

The iris example needs:
- New verdict values in conclude calls
- No promote/discard steps
- Parent support (optional for iris)

- [ ] **Step 2: Run full test suite**

```bash
npm test
```

- [ ] **Step 3: Verify all exports are correct in index.ts**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(research-loop): update examples + verify exports (harness 2 extensions complete)"
```
