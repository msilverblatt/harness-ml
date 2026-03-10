# Notebook Studio UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Notebook tab to Studio with timeline + journal views, integrate notebook narrative into Dashboard and Activity tabs.

**Architecture:** New `Notebook` view with two sub-views (Timeline, Journal), a shared `useNotebook` hook for data fetching, notebook entry type definitions in `api.ts`, new CSS custom properties for entry type colors. Dashboard gets a narrative hero section. Activity gets a pinned context bar and inline notebook entries.

**Tech Stack:** React 19, TypeScript, CSS Modules, existing `useApi`/`useRefreshKey` hooks, `MarkdownRenderer` component, design tokens.

---

### Task 1: TypeScript Types + Notebook Entry Color Tokens

**Files:**
- Modify: `packages/harness-studio/frontend/src/types/api.ts`
- Modify: `packages/harness-studio/frontend/src/styles/tokens.css`

**Step 1: Add notebook types to api.ts**

Add after the `Experiment` types block (after line 138):

```typescript
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
```

**Step 2: Add notebook entry type color tokens to tokens.css**

Add after the tool badge colors block (after line 66):

```css
/* Notebook entry type colors */
--color-nb-theory: #d2a8ff;
--color-nb-finding: #58a6ff;
--color-nb-research: #56d364;
--color-nb-decision: #f0883e;
--color-nb-plan: #3fb950;
--color-nb-note: #6e7681;

--color-nb-theory-a15: rgba(210, 168, 255, 0.15);
--color-nb-finding-a15: rgba(88, 166, 255, 0.15);
--color-nb-research-a15: rgba(86, 211, 100, 0.15);
--color-nb-decision-a15: rgba(240, 136, 62, 0.15);
--color-nb-plan-a15: rgba(63, 185, 80, 0.15);
--color-nb-note-a15: rgba(110, 118, 129, 0.15);
```

**Step 3: Verify TypeScript compiles**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: No new errors

**Step 4: Commit**

```bash
git add packages/harness-studio/frontend/src/types/api.ts packages/harness-studio/frontend/src/styles/tokens.css
git commit -m "feat(studio): add notebook TypeScript types and color tokens"
```

---

### Task 2: Notebook Entry Components (shared between views)

**Files:**
- Create: `packages/harness-studio/frontend/src/components/NotebookEntry/NotebookEntry.tsx`
- Create: `packages/harness-studio/frontend/src/components/NotebookEntry/NotebookEntry.module.css`

These are the reusable building blocks used by Timeline, Journal, Dashboard, and Activity.

**Step 1: Create NotebookEntry.module.css**

```css
.entryCard {
    padding: var(--space-3) var(--space-4);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg-secondary);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}

.entryCard:hover {
    border-color: var(--color-border-subtle);
    background: var(--color-bg-tertiary);
}

.entryCardStruck {
    opacity: 0.5;
}

.entryCardStruck .entryContent {
    text-decoration: line-through;
}

.entryMeta {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
}

.typeBadge {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    font-weight: 600;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.timestamp {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    cursor: default;
}

.experimentLink {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-accent);
    text-decoration: none;
    padding: 1px 4px;
    border-radius: var(--radius-sm);
    background: var(--color-accent-a08);
}

.experimentLink:hover {
    background: var(--color-accent-a25);
}

.entryContent {
    font-size: var(--font-size-sm);
    line-height: 1.5;
    color: var(--color-text-primary);
}

.tagList {
    display: flex;
    gap: var(--space-1);
    flex-wrap: wrap;
}

.tag {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--color-text-secondary);
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    padding: 0 4px;
    border-radius: var(--radius-sm);
    cursor: pointer;
}

.tag:hover {
    border-color: var(--color-accent);
    color: var(--color-accent);
}

.struckReason {
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    font-style: italic;
}

.struckIndicator {
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    cursor: pointer;
    padding: var(--space-1) 0;
}

.struckIndicator:hover {
    color: var(--color-text-secondary);
}

/* Compact variant for dashboard/activity inline display */
.entryCompact {
    padding: var(--space-2) var(--space-3);
    border: none;
    border-left: 3px solid var(--color-border);
    border-radius: 0;
    background: transparent;
}

.entryCompact .entryContent {
    font-size: var(--font-size-xs);
    max-height: 3em;
    overflow: hidden;
}
```

**Step 2: Create NotebookEntry.tsx**

```tsx
import { Link, useParams } from 'react-router-dom';
import { MarkdownRenderer } from '../MarkdownRenderer/MarkdownRenderer';
import { humanizeTimestamp, formatTimestamp } from '../../utils/time';
import type { NotebookEntry as NotebookEntryType, NotebookEntryType as EntryTypeEnum } from '../../types/api';
import styles from './NotebookEntry.module.css';

const TYPE_COLORS: Record<EntryTypeEnum, { bg: string; fg: string }> = {
    theory: { bg: 'var(--color-nb-theory-a15)', fg: 'var(--color-nb-theory)' },
    finding: { bg: 'var(--color-nb-finding-a15)', fg: 'var(--color-nb-finding)' },
    research: { bg: 'var(--color-nb-research-a15)', fg: 'var(--color-nb-research)' },
    decision: { bg: 'var(--color-nb-decision-a15)', fg: 'var(--color-nb-decision)' },
    plan: { bg: 'var(--color-nb-plan-a15)', fg: 'var(--color-nb-plan)' },
    note: { bg: 'var(--color-nb-note-a15)', fg: 'var(--color-nb-note)' },
};

interface NotebookEntryCardProps {
    entry: NotebookEntryType;
    compact?: boolean;
    onTagClick?: (tag: string) => void;
}

export function TypeBadge({ type }: { type: EntryTypeEnum }) {
    const color = TYPE_COLORS[type] ?? TYPE_COLORS.note;
    return (
        <span
            className={styles.typeBadge}
            style={{ backgroundColor: color.bg, color: color.fg }}
        >
            {type}
        </span>
    );
}

export function NotebookEntryCard({ entry, compact, onTagClick }: NotebookEntryCardProps) {
    const { project } = useParams<{ project: string }>();
    const allTags = [...entry.tags, ...entry.auto_tags];
    const cardClass = [
        styles.entryCard,
        compact && styles.entryCompact,
        entry.struck && styles.entryCardStruck,
    ].filter(Boolean).join(' ');

    const borderColor = TYPE_COLORS[entry.type]?.fg ?? 'var(--color-border)';

    return (
        <div className={cardClass} style={compact ? { borderLeftColor: borderColor } : undefined}>
            <div className={styles.entryMeta}>
                <TypeBadge type={entry.type} />
                <span className={styles.timestamp} title={formatTimestamp(entry.timestamp)}>
                    {humanizeTimestamp(entry.timestamp)}
                </span>
                {entry.experiment_id && (
                    <Link
                        to={`/${project}/experiments`}
                        state={{ expandedId: entry.experiment_id }}
                        className={styles.experimentLink}
                    >
                        {entry.experiment_id}
                    </Link>
                )}
                {entry.struck && entry.struck_reason && (
                    <span className={styles.struckReason}>struck: {entry.struck_reason}</span>
                )}
            </div>
            <div className={styles.entryContent}>
                <MarkdownRenderer content={entry.content} />
            </div>
            {allTags.length > 0 && !compact && (
                <div className={styles.tagList}>
                    {allTags.map(tag => (
                        <span
                            key={tag}
                            className={styles.tag}
                            onClick={() => onTagClick?.(tag)}
                        >
                            {tag}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}

interface StruckIndicatorProps {
    count: number;
    expanded: boolean;
    onToggle: () => void;
}

export function StruckIndicator({ count, expanded, onToggle }: StruckIndicatorProps) {
    if (count === 0) return null;
    return (
        <div className={styles.struckIndicator} onClick={onToggle}>
            {expanded ? 'Hide' : 'Show'} {count} previous {count === 1 ? 'version' : 'versions'}
        </div>
    );
}
```

**Step 3: Verify TypeScript compiles**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: No new errors

**Step 4: Commit**

```bash
git add packages/harness-studio/frontend/src/components/NotebookEntry/
git commit -m "feat(studio): add NotebookEntry shared components"
```

---

### Task 3: Notebook View — Timeline + Journal + Filters

**Files:**
- Create: `packages/harness-studio/frontend/src/views/Notebook/Notebook.tsx`
- Create: `packages/harness-studio/frontend/src/views/Notebook/Notebook.module.css`
- Create: `packages/harness-studio/frontend/src/views/Notebook/TimelineView.tsx`
- Create: `packages/harness-studio/frontend/src/views/Notebook/JournalView.tsx`
- Create: `packages/harness-studio/frontend/src/views/Notebook/FilterBar.tsx`

**Step 1: Create Notebook.module.css**

```css
.notebook {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-4);
    height: 100%;
    overflow-y: auto;
}

.viewToggle {
    display: flex;
    gap: 0;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    overflow: hidden;
    width: fit-content;
}

.viewButton {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    padding: var(--space-1) var(--space-3);
    background: transparent;
    color: var(--color-text-secondary);
    border: none;
    cursor: pointer;
}

.viewButton:hover {
    background: var(--color-bg-hover);
}

.viewButtonActive {
    background: var(--color-accent-a12);
    color: var(--color-accent);
}

.topBar {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
}

/* Filter bar */
.filterBar {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
}

.filterSelect {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    padding: var(--space-1) var(--space-2);
    background: var(--color-bg-secondary);
    color: var(--color-text-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
}

.filterInput {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    padding: var(--space-1) var(--space-2);
    background: var(--color-bg-secondary);
    color: var(--color-text-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    min-width: 160px;
}

.filterInput::placeholder {
    color: var(--color-text-muted);
}

.dateInput {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    padding: var(--space-1) var(--space-2);
    background: var(--color-bg-secondary);
    color: var(--color-text-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
}

.filterCount {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    margin-left: auto;
}

.clearButton {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    padding: var(--space-1) var(--space-2);
    background: transparent;
    color: var(--color-text-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: pointer;
}

.clearButton:hover {
    color: var(--color-text-primary);
    border-color: var(--color-text-muted);
}

.struckToggle {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    padding: var(--space-1) var(--space-2);
    background: transparent;
    color: var(--color-text-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: pointer;
}

.struckToggleActive {
    background: var(--color-accent-a08);
    color: var(--color-accent);
    border-color: var(--color-accent-a25);
}

/* Timeline view */
.timeline {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
}

/* Journal view */
.journalSection {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
}

.journalSectionHeader {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding-bottom: var(--space-1);
    border-bottom: 1px solid var(--color-border);
}

.heroCards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
}

@media (max-width: 768px) {
    .heroCards {
        grid-template-columns: 1fr;
    }
}

.heroCard {
    padding: var(--space-4);
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
    background: var(--color-bg-secondary);
}

.heroCardLabel {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: var(--space-2);
    display: flex;
    align-items: center;
    gap: var(--space-2);
}

.heroCardContent {
    font-size: var(--font-size-sm);
    line-height: 1.5;
    color: var(--color-text-primary);
}

.experimentGroup {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}

.experimentGroupHeader {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-accent);
    padding: var(--space-1) 0;
}

.emptyState {
    text-align: center;
    padding: var(--space-8) var(--space-4);
    color: var(--color-text-muted);
    font-size: var(--font-size-sm);
}
```

**Step 2: Create FilterBar.tsx**

```tsx
import type { NotebookEntry, NotebookEntryType } from '../../types/api';
import styles from './Notebook.module.css';

const ENTRY_TYPES: NotebookEntryType[] = ['theory', 'finding', 'research', 'decision', 'plan', 'note'];

export interface NotebookFilters {
    type: string;
    text: string;
    tag: string;
    experimentId: string;
    dateFrom: string;
    dateTo: string;
    showStruck: boolean;
}

export const EMPTY_FILTERS: NotebookFilters = {
    type: '',
    text: '',
    tag: '',
    experimentId: '',
    dateFrom: '',
    dateTo: '',
    showStruck: false,
};

interface FilterBarProps {
    filters: NotebookFilters;
    onChange: (filters: NotebookFilters) => void;
    entries: NotebookEntry[];
    filteredCount: number;
}

export function filterEntries(entries: NotebookEntry[], filters: NotebookFilters): NotebookEntry[] {
    return entries.filter(entry => {
        if (!filters.showStruck && entry.struck) return false;
        if (filters.type && entry.type !== filters.type) return false;
        if (filters.text) {
            const search = filters.text.toLowerCase();
            const haystack = `${entry.content} ${entry.tags.join(' ')} ${entry.auto_tags.join(' ')}`.toLowerCase();
            if (!haystack.includes(search)) return false;
        }
        if (filters.tag) {
            const allTags = [...entry.tags, ...entry.auto_tags];
            if (!allTags.includes(filters.tag)) return false;
        }
        if (filters.experimentId && entry.experiment_id !== filters.experimentId) return false;
        if (filters.dateFrom) {
            const entryDate = new Date(entry.timestamp).toISOString().slice(0, 10);
            if (entryDate < filters.dateFrom) return false;
        }
        if (filters.dateTo) {
            const entryDate = new Date(entry.timestamp).toISOString().slice(0, 10);
            if (entryDate > filters.dateTo) return false;
        }
        return true;
    });
}

export function FilterBar({ filters, onChange, entries, filteredCount }: FilterBarProps) {
    const allTags = Array.from(
        new Set(entries.flatMap(e => [...e.tags, ...e.auto_tags]))
    ).sort();

    const allExperimentIds = Array.from(
        new Set(entries.map(e => e.experiment_id).filter((id): id is string => id !== null))
    ).sort();

    const struckCount = entries.filter(e => e.struck).length;

    const hasActiveFilters = filters.type || filters.text || filters.tag
        || filters.experimentId || filters.dateFrom || filters.dateTo;

    function set<K extends keyof NotebookFilters>(key: K, value: NotebookFilters[K]) {
        onChange({ ...filters, [key]: value });
    }

    return (
        <div className={styles.filterBar}>
            <select
                className={styles.filterSelect}
                value={filters.type}
                onChange={e => set('type', e.target.value)}
            >
                <option value="">All types</option>
                {ENTRY_TYPES.map(t => (
                    <option key={t} value={t}>{t}</option>
                ))}
            </select>
            <input
                className={styles.filterInput}
                type="text"
                placeholder="Search entries..."
                value={filters.text}
                onChange={e => set('text', e.target.value)}
            />
            {allTags.length > 0 && (
                <select
                    className={styles.filterSelect}
                    value={filters.tag}
                    onChange={e => set('tag', e.target.value)}
                >
                    <option value="">All tags</option>
                    {allTags.map(t => (
                        <option key={t} value={t}>{t}</option>
                    ))}
                </select>
            )}
            {allExperimentIds.length > 0 && (
                <select
                    className={styles.filterSelect}
                    value={filters.experimentId}
                    onChange={e => set('experimentId', e.target.value)}
                >
                    <option value="">All experiments</option>
                    {allExperimentIds.map(id => (
                        <option key={id} value={id}>{id}</option>
                    ))}
                </select>
            )}
            <input
                className={styles.dateInput}
                type="date"
                value={filters.dateFrom}
                onChange={e => set('dateFrom', e.target.value)}
                title="From date"
            />
            <input
                className={styles.dateInput}
                type="date"
                value={filters.dateTo}
                onChange={e => set('dateTo', e.target.value)}
                title="To date"
            />
            {struckCount > 0 && (
                <button
                    className={`${styles.struckToggle} ${filters.showStruck ? styles.struckToggleActive : ''}`}
                    onClick={() => set('showStruck', !filters.showStruck)}
                >
                    {struckCount} struck
                </button>
            )}
            <span className={styles.filterCount}>
                {filteredCount} of {entries.filter(e => !e.struck || filters.showStruck).length}
            </span>
            {hasActiveFilters && (
                <button
                    className={styles.clearButton}
                    onClick={() => onChange({ ...EMPTY_FILTERS, showStruck: filters.showStruck })}
                >
                    Clear
                </button>
            )}
        </div>
    );
}
```

**Step 3: Create TimelineView.tsx**

```tsx
import { useState, useMemo } from 'react';
import type { NotebookEntry } from '../../types/api';
import { NotebookEntryCard, StruckIndicator } from '../../components/NotebookEntry/NotebookEntry';
import styles from './Notebook.module.css';

interface TimelineViewProps {
    entries: NotebookEntry[];
    onTagClick: (tag: string) => void;
}

export function TimelineView({ entries, onTagClick }: TimelineViewProps) {
    const [expandedStruck, setExpandedStruck] = useState<Set<string>>(new Set());

    // Group entries by ID to find struck versions
    const { visible, struckBySuccessor } = useMemo(() => {
        const struckMap = new Map<string, NotebookEntry[]>();
        const visibleEntries: NotebookEntry[] = [];

        for (const entry of entries) {
            if (entry.struck) {
                // Group struck entries after their non-struck successors in the timeline
                // For now, just collect them — they'll appear inline
                visibleEntries.push(entry);
            } else {
                visibleEntries.push(entry);
            }
        }

        return { visible: visibleEntries, struckBySuccessor: struckMap };
    }, [entries]);

    if (visible.length === 0) {
        return <div className={styles.emptyState}>No notebook entries yet</div>;
    }

    // Count struck entries adjacent to non-struck entries of same type
    const struckCounts = useMemo(() => {
        const counts = new Map<number, number>();
        let i = 0;
        while (i < visible.length) {
            if (!visible[i].struck) {
                // Count struck entries immediately following with same type
                let count = 0;
                let j = i + 1;
                while (j < visible.length && visible[j].struck && visible[j].type === visible[i].type) {
                    count++;
                    j++;
                }
                if (count > 0) {
                    counts.set(i, count);
                }
            }
            i++;
        }
        return counts;
    }, [visible]);

    function toggleStruck(idx: number) {
        setExpandedStruck(prev => {
            const next = new Set(prev);
            const key = String(idx);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    }

    return (
        <div className={styles.timeline}>
            {visible.map((entry, idx) => {
                // Skip struck entries that are grouped under a non-struck entry
                if (entry.struck) {
                    // Find if this is grouped under a predecessor
                    let predecessorIdx = idx - 1;
                    while (predecessorIdx >= 0 && visible[predecessorIdx].struck) {
                        predecessorIdx--;
                    }
                    if (predecessorIdx >= 0
                        && !visible[predecessorIdx].struck
                        && visible[predecessorIdx].type === entry.type
                        && !expandedStruck.has(String(predecessorIdx))
                    ) {
                        return null; // Hidden, part of a struck group
                    }
                }

                const struckCount = struckCounts.get(idx) ?? 0;

                return (
                    <div key={entry.id}>
                        <NotebookEntryCard entry={entry} onTagClick={onTagClick} />
                        {struckCount > 0 && (
                            <StruckIndicator
                                count={struckCount}
                                expanded={expandedStruck.has(String(idx))}
                                onToggle={() => toggleStruck(idx)}
                            />
                        )}
                    </div>
                );
            })}
        </div>
    );
}
```

**Step 4: Create JournalView.tsx**

```tsx
import { useMemo } from 'react';
import type { NotebookEntry } from '../../types/api';
import { NotebookEntryCard, TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import styles from './Notebook.module.css';

interface JournalViewProps {
    entries: NotebookEntry[];
    onTagClick: (tag: string) => void;
}

export function JournalView({ entries, onTagClick }: JournalViewProps) {
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);

    // Latest theory and plan (pinned hero cards)
    const latestTheory = useMemo(
        () => nonStruck.find(e => e.type === 'theory') ?? null,
        [nonStruck]
    );
    const latestPlan = useMemo(
        () => nonStruck.find(e => e.type === 'plan') ?? null,
        [nonStruck]
    );

    // Group remaining entries by experiment_id
    const { grouped, general } = useMemo(() => {
        const remaining = nonStruck.filter(
            e => e.type !== 'theory' && e.type !== 'plan'
                || (e.type === 'theory' && e !== latestTheory)
                || (e.type === 'plan' && e !== latestPlan)
        );

        const byExp = new Map<string, NotebookEntry[]>();
        const noExp: NotebookEntry[] = [];

        for (const entry of remaining) {
            if (entry.experiment_id) {
                const list = byExp.get(entry.experiment_id) ?? [];
                list.push(entry);
                byExp.set(entry.experiment_id, list);
            } else {
                noExp.push(entry);
            }
        }

        // Sort experiment groups by most recent entry
        const sorted = Array.from(byExp.entries()).sort((a, b) => {
            const aTime = a[1][0]?.timestamp ?? '';
            const bTime = b[1][0]?.timestamp ?? '';
            return bTime.localeCompare(aTime);
        });

        return { grouped: sorted, general: noExp };
    }, [nonStruck, latestTheory, latestPlan]);

    if (nonStruck.length === 0) {
        return <div className={styles.emptyState}>No notebook entries yet</div>;
    }

    return (
        <div className={styles.journalSection}>
            {/* Hero cards: latest theory + plan */}
            {(latestTheory || latestPlan) && (
                <div className={styles.heroCards}>
                    {latestTheory && (
                        <div className={styles.heroCard} style={{ borderColor: 'var(--color-nb-theory)' }}>
                            <div className={styles.heroCardLabel}>
                                <TypeBadge type="theory" />
                                Current Theory
                            </div>
                            <div className={styles.heroCardContent}>
                                <MarkdownRenderer content={latestTheory.content} />
                            </div>
                        </div>
                    )}
                    {latestPlan && (
                        <div className={styles.heroCard} style={{ borderColor: 'var(--color-nb-plan)' }}>
                            <div className={styles.heroCardLabel}>
                                <TypeBadge type="plan" />
                                Current Plan
                            </div>
                            <div className={styles.heroCardContent}>
                                <MarkdownRenderer content={latestPlan.content} />
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Entries grouped by experiment */}
            {grouped.map(([expId, entries]) => (
                <div key={expId} className={styles.experimentGroup}>
                    <div className={styles.experimentGroupHeader}>{expId}</div>
                    {entries.map(entry => (
                        <NotebookEntryCard key={entry.id} entry={entry} onTagClick={onTagClick} />
                    ))}
                </div>
            ))}

            {/* General (no experiment) entries */}
            {general.length > 0 && (
                <div className={styles.experimentGroup}>
                    {grouped.length > 0 && (
                        <div className={styles.experimentGroupHeader}>General</div>
                    )}
                    {general.map(entry => (
                        <NotebookEntryCard key={entry.id} entry={entry} onTagClick={onTagClick} />
                    ))}
                </div>
            )}
        </div>
    );
}
```

**Step 5: Create Notebook.tsx (main view)**

```tsx
import { useState, useMemo } from 'react';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useProject } from '../../hooks/useProject';
import { EmptyState } from '../../components/EmptyState/EmptyState';
import { FilterBar, filterEntries, EMPTY_FILTERS, type NotebookFilters } from './FilterBar';
import { TimelineView } from './TimelineView';
import { JournalView } from './JournalView';
import type { NotebookEntry } from '../../types/api';
import styles from './Notebook.module.css';

type ViewMode = 'timeline' | 'journal';

export function Notebook() {
    const { events } = useLayoutContext();
    const project = useProject();
    const nbKey = useRefreshKey(events, ['notebook']);
    const { data: entries, loading, error } = useApi<NotebookEntry[]>('/api/notebook', nbKey, project);
    const [view, setView] = useState<ViewMode>('journal');
    const [filters, setFilters] = useState<NotebookFilters>(EMPTY_FILTERS);

    const allEntries = entries ?? [];
    const filtered = useMemo(() => filterEntries(allEntries, filters), [allEntries, filters]);

    function handleTagClick(tag: string) {
        setFilters(prev => ({ ...prev, tag }));
    }

    if (loading) {
        return (
            <div className={styles.notebook}>
                <div className={styles.emptyState}>Loading notebook...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.notebook}>
                <div className={styles.emptyState}>Error: {error}</div>
            </div>
        );
    }

    if (allEntries.length === 0) {
        return (
            <div className={styles.notebook}>
                <EmptyState
                    title="No notebook entries"
                    description="The agent writes theories, findings, and plans to the notebook as it works. Entries will appear here."
                />
            </div>
        );
    }

    return (
        <div className={styles.notebook}>
            <div className={styles.topBar}>
                <div className={styles.viewToggle}>
                    <button
                        className={`${styles.viewButton} ${view === 'journal' ? styles.viewButtonActive : ''}`}
                        onClick={() => setView('journal')}
                    >
                        Journal
                    </button>
                    <button
                        className={`${styles.viewButton} ${view === 'timeline' ? styles.viewButtonActive : ''}`}
                        onClick={() => setView('timeline')}
                    >
                        Timeline
                    </button>
                </div>
                <FilterBar
                    filters={filters}
                    onChange={setFilters}
                    entries={allEntries}
                    filteredCount={filtered.length}
                />
            </div>
            {view === 'timeline' ? (
                <TimelineView entries={filtered} onTagClick={handleTagClick} />
            ) : (
                <JournalView entries={filtered} onTagClick={handleTagClick} />
            )}
        </div>
    );
}
```

**Step 6: Verify TypeScript compiles**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: No new errors

**Step 7: Commit**

```bash
git add packages/harness-studio/frontend/src/views/Notebook/
git commit -m "feat(studio): add Notebook view with timeline and journal modes"
```

---

### Task 4: Register Notebook Route + Sidebar Nav

**Files:**
- Modify: `packages/harness-studio/frontend/src/App.tsx`
- Modify: `packages/harness-studio/frontend/src/components/Layout/Layout.tsx`

**Step 1: Add route in App.tsx**

Add import at top (after line 21):

```typescript
import { Notebook } from './views/Notebook/Notebook';
```

Add route after the experiments route (after line 49):

```tsx
<Route path="notebook" element={<ErrorBoundary><Notebook /></ErrorBoundary>} />
```

**Step 2: Add sidebar nav in Layout.tsx**

Add a notebook icon function after `IconExperiments` (after line 184):

```tsx
function IconNotebook() {
    return <svg {...I}><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/><line x1="8" y1="7" x2="16" y2="7"/><line x1="8" y1="11" x2="13" y2="11"/></svg>;
}
```

Add the Notebook tab to the Agent section (after the Experiments entry, around line 309):

```typescript
{ path: `/${projectName}/notebook`, label: 'Notebook', icon: <IconNotebook /> },
```

**Step 3: Add keyboard shortcut for notebook**

In the `shortcutHandlers` callback (around line 289), add:

```typescript
notebook: () => navigate(`/${projectName}/notebook`),
```

**Step 4: Verify TypeScript compiles and dev server runs**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: No new errors

**Step 5: Commit**

```bash
git add packages/harness-studio/frontend/src/App.tsx packages/harness-studio/frontend/src/components/Layout/Layout.tsx
git commit -m "feat(studio): register Notebook route and sidebar nav"
```

---

### Task 5: Dashboard Narrative Hero Section

**Files:**
- Modify: `packages/harness-studio/frontend/src/views/Dashboard/Dashboard.tsx`
- Modify: `packages/harness-studio/frontend/src/views/Dashboard/Dashboard.module.css`

**Step 1: Add narrative widget component in Dashboard.tsx**

Add after the existing imports:

```typescript
import type { NotebookEntry } from '../../types/api';
import { TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
```

Add a new widget component before the `Dashboard` export function:

```tsx
function NarrativeWidget({ entries }: { entries: NotebookEntry[] }) {
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);
    const latestTheory = nonStruck.find(e => e.type === 'theory') ?? null;
    const latestPlan = nonStruck.find(e => e.type === 'plan') ?? null;
    const recentFindings = nonStruck.filter(e => e.type === 'finding').slice(0, 3);

    if (!latestTheory && !latestPlan && recentFindings.length === 0) {
        return null;
    }

    return (
        <div className={`${styles.widget} ${styles.narrativeWidget}`}>
            <div className={styles.widgetHeader}>
                <span className={styles.widgetTitle}>Agent Narrative</span>
                <Link to="../notebook" className={styles.widgetLink}>Full notebook</Link>
            </div>
            <div className={styles.widgetBody}>
                <div className={styles.narrativeGrid}>
                    {latestTheory && (
                        <div className={styles.narrativeCard}>
                            <div className={styles.narrativeLabel}>
                                <TypeBadge type="theory" /> Current Theory
                            </div>
                            <div className={styles.narrativeContent}>
                                <MarkdownRenderer content={latestTheory.content} />
                            </div>
                        </div>
                    )}
                    {latestPlan && (
                        <div className={styles.narrativeCard}>
                            <div className={styles.narrativeLabel}>
                                <TypeBadge type="plan" /> Current Plan
                            </div>
                            <div className={styles.narrativeContent}>
                                <MarkdownRenderer content={latestPlan.content} />
                            </div>
                        </div>
                    )}
                </div>
                {recentFindings.length > 0 && (
                    <div className={styles.narrativeFindings}>
                        <div className={styles.narrativeFindingsLabel}>Recent Findings</div>
                        {recentFindings.map(f => (
                            <div key={f.id} className={styles.narrativeFinding}>
                                <TypeBadge type="finding" />
                                <div className={styles.narrativeFindingContent}>
                                    <MarkdownRenderer content={f.content} />
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
```

**Step 2: Fetch notebook data and render in Dashboard export**

In the `Dashboard` component, add after the existing `useApi` calls:

```typescript
const nbKey = useRefreshKey(events, ['notebook']);
const { data: notebookEntries } = useApi<NotebookEntry[]>('/api/notebook', nbKey, project);
const nbEntries = notebookEntries ?? [];
```

In the return JSX, add `NarrativeWidget` as the first child of the dashboard div:

```tsx
<NarrativeWidget entries={nbEntries} />
```

**Step 3: Add CSS for narrative widget in Dashboard.module.css**

Add at the end of the file:

```css
.narrativeWidget {
    grid-column: 1 / -1;
    order: -1;
}

.narrativeGrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
}

@media (max-width: 768px) {
    .narrativeGrid {
        grid-template-columns: 1fr;
    }
}

.narrativeCard {
    padding: var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg-primary);
}

.narrativeLabel {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
}

.narrativeContent {
    font-size: var(--font-size-sm);
    line-height: 1.5;
    color: var(--color-text-primary);
    max-height: 8em;
    overflow: hidden;
}

.narrativeFindings {
    margin-top: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}

.narrativeFindingsLabel {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.narrativeFinding {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-2);
    border-left: 2px solid var(--color-nb-finding);
    background: var(--color-nb-finding-a15);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

.narrativeFindingContent {
    font-size: var(--font-size-xs);
    line-height: 1.4;
    color: var(--color-text-primary);
    max-height: 3em;
    overflow: hidden;
}
```

**Step 4: Verify TypeScript compiles**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: No new errors

**Step 5: Commit**

```bash
git add packages/harness-studio/frontend/src/views/Dashboard/
git commit -m "feat(studio): add narrative hero section to Dashboard"
```

---

### Task 6: Activity Tab — Context Bar + Inline Notebook Entries

**Files:**
- Modify: `packages/harness-studio/frontend/src/views/Activity/Activity.tsx`
- Modify: `packages/harness-studio/frontend/src/views/Activity/Activity.module.css`

**Step 1: Create ContextBar component in Activity.tsx**

Replace the content of `Activity.tsx`:

```tsx
import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useLayoutContext } from '../../components/Layout/Layout';
import { useApi } from '../../hooks/useApi';
import { useRefreshKey } from '../../hooks/useRefreshKey';
import { useProject } from '../../hooks/useProject';
import { TypeBadge } from '../../components/NotebookEntry/NotebookEntry';
import { MarkdownRenderer } from '../../components/MarkdownRenderer/MarkdownRenderer';
import type { NotebookEntry, Experiment } from '../../types/api';
import { StatBar } from './StatBar';
import { EventLog } from './EventLog';
import styles from './Activity.module.css';

function ContextBar({ entries, experiments }: { entries: NotebookEntry[]; experiments: Experiment[] }) {
    const { project } = useParams<{ project: string }>();
    const nonStruck = useMemo(() => entries.filter(e => !e.struck), [entries]);
    const latestTheory = nonStruck.find(e => e.type === 'theory') ?? null;
    const latestPlan = nonStruck.find(e => e.type === 'plan') ?? null;

    // Latest or running experiment
    const latestExperiment = experiments.length > 0 ? experiments[0] : null;

    if (!latestTheory && !latestPlan && !latestExperiment) return null;

    return (
        <div className={styles.contextBar}>
            {latestTheory && (
                <div className={styles.contextItem}>
                    <TypeBadge type="theory" />
                    <div className={styles.contextText}>
                        <MarkdownRenderer content={latestTheory.content} />
                    </div>
                </div>
            )}
            {latestPlan && (
                <div className={styles.contextItem}>
                    <TypeBadge type="plan" />
                    <div className={styles.contextText}>
                        <MarkdownRenderer content={latestPlan.content} />
                    </div>
                </div>
            )}
            {latestExperiment && (
                <div className={styles.contextItem}>
                    <span className={styles.contextExpBadge}>{latestExperiment.experiment_id}</span>
                    <div className={styles.contextText}>
                        {latestExperiment.hypothesis ?? latestExperiment.description ?? '--'}
                    </div>
                    {latestExperiment.verdict && (
                        <span className={styles.contextVerdict}>{latestExperiment.verdict}</span>
                    )}
                </div>
            )}
            <Link to={`/${project}/notebook`} className={styles.contextLink}>
                Notebook
            </Link>
        </div>
    );
}

export function Activity() {
    const { events } = useLayoutContext();
    const project = useProject();
    const nbKey = useRefreshKey(events, ['notebook']);
    const expKey = useRefreshKey(events, ['experiments', 'pipeline']);
    const { data: notebookEntries } = useApi<NotebookEntry[]>('/api/notebook', nbKey, project);
    const { data: experiments } = useApi<Experiment[]>('/api/experiments', expKey, project);

    return (
        <div className={styles.activity}>
            <ContextBar entries={notebookEntries ?? []} experiments={experiments ?? []} />
            <StatBar />
            <EventLog wsEvents={events} />
        </div>
    );
}
```

**Step 2: Add CSS for context bar in Activity.module.css**

Add at the end of the file:

```css
/* Context Bar */
.contextBar {
    display: flex;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: var(--color-bg-secondary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    align-items: flex-start;
    flex-wrap: wrap;
}

.contextItem {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    flex: 1;
    min-width: 200px;
}

.contextText {
    font-size: var(--font-size-xs);
    line-height: 1.4;
    color: var(--color-text-primary);
    max-height: 3em;
    overflow: hidden;
}

.contextExpBadge {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    font-weight: 600;
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: var(--color-orange-a15);
    color: var(--color-orange);
    white-space: nowrap;
}

.contextVerdict {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 1px 4px;
    border-radius: var(--radius-sm);
    background: var(--color-bg-tertiary);
    color: var(--color-text-secondary);
    white-space: nowrap;
}

.contextLink {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    color: var(--color-accent);
    text-decoration: none;
    white-space: nowrap;
    align-self: center;
}

.contextLink:hover {
    text-decoration: underline;
}
```

**Step 3: Verify TypeScript compiles**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: No new errors

**Step 4: Commit**

```bash
git add packages/harness-studio/frontend/src/views/Activity/
git commit -m "feat(studio): add context bar with narrative to Activity tab"
```

---

### Task 7: Build Frontend + Verify

**Files:**
- No new files — build + verify pipeline

**Step 1: Run TypeScript check**

Run: `cd packages/harness-studio/frontend && npx tsc --noEmit`
Expected: PASS

**Step 2: Build frontend**

Run: `bash packages/harness-studio/scripts/build_frontend.sh`
Expected: Build succeeds, static assets copied

**Step 3: Run existing Studio tests**

Run: `uv run pytest packages/harness-studio/tests/ -v`
Expected: All existing tests pass

**Step 4: Commit built assets**

```bash
git add packages/harness-studio/src/harnessml/studio/static/
git commit -m "build(studio): rebuild frontend with notebook UI"
```
