# Workspace locking and recovery design

## Status

Approved as the second Step 1 roadmap milestone. This milestone covers mutation
serialization, complete-before-visible version publication, and deterministic
recovery. Cooperative cancellation and resource budgets remain follow-up designs.

## Problem

Harness workspace mutations currently assume one process. Two agents can choose
the same next version, overwrite live configuration, or race the `current`
pointer. A process terminated between version creation, artifact writing, config
restoration, and pointer update can also leave a visible partial version or live
configuration inconsistent with `current`.

## Goals

- Permit at most one workspace mutation across processes.
- Fail concurrent mutation attempts quickly with actionable owner metadata.
- Keep long-running training output hidden until the complete version is ready.
- Recover deterministically after termination at any commit step.
- Keep reads available while training is running.
- Preserve the existing version layout and APIs where practical.

## Non-goals

- Distributed scheduling or a remote lock service.
- Cooperative cancellation of an active trainer.
- CPU, memory, GPU, time, or monetary enforcement.
- Transactional mutation of external data sources.
- Lock-free reads with a full multi-version configuration store.

## Workspace lock

All state-changing `WorkspaceManager` operations acquire an exclusive file lock at
`.harness/workspace.lock`. The default timeout is zero: a second mutation fails
rather than waiting indefinitely behind a training run. Programmatic callers may
configure a positive timeout.

While held, `.harness/workspace-lock.json` contains:

- process ID;
- hostname;
- operation name;
- acquisition timestamp.

The metadata is diagnostic only; OS-backed file locking determines ownership.
A terminated process releases the OS lock automatically. The next owner replaces
stale metadata.

Covered operations:

- experiment execution and promotion;
- version switching;
- conclusion/verdict updates.

Read-only status, comparison, ancestry, diagnostics, and artifact inspection do
not acquire the exclusive lock.

## Complete-before-visible publication

Experiment construction uses two hidden staging locations:

1. `.experiment-*` for the mutable candidate configuration;
2. `versions/.<version>.<uuid>.tmp` for metadata, config snapshot, run state,
   predictions, diagnostics, evals, explanations, and production bundle.

The version staging directory contains `run/state.json`. It begins as `running`
and becomes `complete` only after all run artifacts are durable. Only then is the
whole directory atomically renamed to `versions/<version>`.

A normal reader therefore sees either no candidate version or a complete version,
never a version whose run artifacts are still being written.

## Commit journal

The final workspace promotion spans multiple filesystem objects and cannot be one
rename. Before publication, Harness atomically writes
`.harness/transaction.json` with:

- transaction ID;
- operation;
- candidate version;
- previous current version;
- timestamp.

Commit order:

1. write the transaction journal;
2. atomically publish the complete version directory;
3. restore the candidate configuration into live `config/`;
4. atomically replace the `current` pointer;
5. remove the journal.

## Recovery

After acquiring the lock and before any mutation, Harness performs recovery.

If a transaction journal exists:

- when `current` already names the candidate, the commit is considered complete;
  candidate config is restored to ensure consistency and the journal is cleared;
- otherwise the commit is rolled back: candidate version is removed, previous
  config/current are restored, and the journal is cleared.

After journal recovery, abandoned `.experiment-*` and hidden version staging
directories are removed. Cleanup occurs only while holding the workspace lock, so
an active owner’s staging files cannot be mistaken for abandoned work.

## Atomic files

Lock metadata, transaction journals, run state, mutable version metadata, and the
`current` pointer use sibling temporary files followed by `replace`. On POSIX,
files and parent directories are flushed where durability matters.

## Failure semantics

- Validation and training failures publish no version and do not change live state.
- Artifact-writing failures remove hidden staging and do not change live state.
- Ordinary exceptions during commit trigger immediate journal recovery.
- Process termination is recovered by the next mutating operation.
- A lock timeout raises `WorkspaceBusyError` including available owner metadata.

## Validation

Tests must cover:

- a second process/thread cannot mutate an occupied workspace;
- lock owner diagnostics and stale metadata replacement;
- versions remain invisible while run artifacts are being written;
- successful publication includes complete run state;
- recovery before publication;
- recovery after publication but before pointer update;
- recovery after pointer update but before journal deletion;
- abandoned staging cleanup;
- atomic current and metadata replacement;
- existing rollback, branching, and real experiment workflows.
