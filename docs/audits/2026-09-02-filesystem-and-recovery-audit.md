# Filesystem, locking, and recovery audit

Date: 2026-09-02

Scope: the first two post-v2 hardening milestones: production bundles and
workspace transaction safety.

## Conclusion

The changes are directionally sound and remain hardening rather than product
expansion, but the audit found one high-severity coverage gap and several limits
that must remain explicit. The high-severity gap—data mutations bypassing the
workspace lock—is fixed by this audit. The system is suitable for continued
hardening, but it is not yet evidence for network-filesystem support or arbitrary
untrusted artifact loading.

## Findings

### H1 — Data mutations bypassed the workspace lock

**Severity:** High  
**Status:** Fixed

`WorkspaceManager.data` exposed a standalone `DataWorkspace`. MCP and Python
callers could mutate source configuration, transform configuration, or the clean
dataset while an experiment held the workspace lock. An experiment could
therefore hash and train against a file concurrently replaced by a pipeline run.

`DataWorkspace` now accepts an optional mutation guard. Instances owned by
`WorkspaceManager` route initialization, source changes, transform changes, and
pipeline execution through the same cross-process workspace lock. The independent
`harness-data` package remains usable without an application workspace.

### H2 — Data configuration writes were not atomic

**Severity:** High  
**Status:** Fixed

`sources.yaml` and `transforms.yaml` used direct writes. Process termination could
leave truncated YAML. They now use flushed temporary files and atomic replacement.
The same primitive is used for application configuration files.

### H3 — Pipeline outputs were written directly

**Severity:** High  
**Status:** Fixed with a documented residual limit

The clean Parquet file and schema were written directly. They are now fully
constructed in hidden files and individually atomically replaced. Abandoned
pipeline staging files are cleaned under the mutation lock. `load_schema()` also
verifies that its declared data hash matches the current Parquet artifact and
fails closed if the pair is inconsistent.

A dataset and its sidecar schema are still two filesystem entries; standard
filesystems do not provide a two-file atomic swap. There is a very short interval
between replacements. Readers receive either an atomic old/new Parquet file, and
schema consumers detect rather than accept a mismatched pair. A future generation
pointer may eliminate this residual interval if operational evidence warrants the
additional layout complexity.

### M1 — Crash recovery tests model states rather than killing every boundary

**Severity:** Medium  
**Status:** Open; required before Step 1 exit

Tests cover lock exclusion and each journal interpretation, but do not yet inject
`SIGKILL` after every filesystem operation. A subprocess fault-injection matrix
must validate actual termination behavior before the roadmap's operational-proof
exit gate can pass.

### M2 — Filesystem support boundary is local filesystems

**Severity:** Medium  
**Status:** Documented limitation

Atomic rename, directory `fsync`, and advisory lock behavior are appropriate for
supported local filesystems. NFS, SMB, object-store mounts, and container volumes
with unusual locking semantics are not validated. Harness must not claim those as
supported until integration tests run against them. A remote coordination and
artifact-store design would be preferable to assuming POSIX semantics remotely.

### M3 — Bundle checksums are integrity checks, not authenticity

**Severity:** Medium  
**Status:** Explicit by design

An attacker can replace both payload and checksum. `trusted=True` is therefore an
acknowledgment of provenance, not a sandbox. Remote retrieval must eventually add
signature or attestation verification before trusted loading.

### M4 — Bundle compatibility metadata is diagnostic

**Severity:** Medium  
**Status:** Open roadmap work

Format compatibility is enforced, but package-version compatibility is recorded
rather than rejected. Strict policy should be based on cross-release fixtures and
real compatibility evidence to avoid arbitrary version rules.

### L1 — Lock metadata is advisory diagnostics

**Severity:** Low  
**Status:** Accepted

The owner JSON can be stale after abrupt termination. It is never used to decide
ownership; the OS-backed lock is authoritative, and the next owner replaces stale
metadata.

## Complexity review

The hardening introduces two state machines:

1. production bundle validation: detect → inspect → verify → trusted deserialize;
2. workspace commit: stage → journal → publish → restore config → switch pointer →
   clear journal.

Both have explicit invariants, narrow modules, failure tests, and documented
legacy behavior. No scheduler, background daemon, distributed lock, generalized
workflow engine, or remote store was introduced. Cancellation and budgets are
intentionally not layered onto these mechanisms until fault-injection work proves
them.

## Required follow-up evidence

Before Step 1 is considered complete:

- run real subprocess `SIGKILL` fault injection across commit and pipeline stages;
- add sustained concurrent read/mutate stress tests;
- establish and test the supported filesystem matrix;
- test bundle fixtures across at least two released Harness versions;
- add authenticity verification before loading remotely sourced bundles;
- measure lock contention and recovery behavior in real projects.
