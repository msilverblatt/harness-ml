# Versioned production bundle design

## Status

Approved for the first Step 1 roadmap implementation.

## Problem

Harness 2.0 writes `ProductionBundle` directly with `cloudpickle`. The file has no
format identifier, compatibility metadata, input contract, or integrity check.
Loading it also executes pickle deserialization without requiring the caller to
acknowledge that the artifact must be trusted.

## Goals

- Preserve the existing single-file `model.bundle` deployment experience.
- Make metadata inspectable without deserializing executable model state.
- Detect corruption and unsupported formats before deserialization.
- Record enough environment and model metadata to diagnose compatibility.
- Require explicit trust at every pickle-loading boundary.
- Read trusted legacy v2.0 bundles during a documented migration period.
- Keep writes atomic.

## Non-goals

- A safe format for loading models supplied by an attacker. Pickle payloads remain
  executable and must be trusted.
- Universal portable model conversion. ONNX and `skops` adapters are later work.
- Exact environment recreation or dependency installation.
- Cryptographic proof of publisher identity. Signing is later work.

## Container

A version 1 bundle is a ZIP container with exactly two required members:

- `manifest.json` — UTF-8 JSON metadata;
- `payload.pkl` — the cloudpickle-serialized `ProductionBundle`.

The public filename remains `model.bundle`. Writers create the complete archive at
a sibling temporary path, flush it, and atomically replace the destination.
Readers never extract archive members to disk.

## Manifest version 1

Required fields:

- `artifact_type`: `harness.production_bundle`;
- `format_version`: integer `1`;
- `created_at`: timezone-aware UTC timestamp;
- `payload`: member name, byte size, and SHA-256 digest;
- `runtime`: Python implementation/version and platform;
- `packages`: versions of Harness and key serialization/runtime dependencies;
- `task`: task type, target column, ensemble method, calibration method, and
  conformal interval availability;
- `models`: model names, model types, seed counts, dependency declarations, and
  ensemble participation;
- `training_features`: ordered feature names and observed pandas dtypes;
- `ensemble_columns`: ordered production ensemble input columns;
- `fingerprints`: canonical project/model/ensemble/feature configuration hashes
  and a training-data hash when available;
- `output`: scalar or per-class prediction contract and interval support.

Unknown fields must be ignored so compatible additions do not require a format
revision. A breaking interpretation change increments `format_version`.

## Loading and trust

`ProductionBundle.inspect(path)` reads and validates only the container and
manifest. It must not import or deserialize the payload.

`ProductionBundle.load(path, trusted=True)` performs these steps:

1. Require `trusted=True`; otherwise fail with a message explaining pickle risk.
2. Detect a versioned ZIP container or a legacy raw-pickle bundle.
3. For a versioned bundle, validate artifact type, supported format version,
   required members, declared payload size, and SHA-256 digest.
4. Only after successful validation, deserialize `payload.pkl`.
5. Verify that the result is a `ProductionBundle`.

Legacy raw-pickle loading is accepted only with `trusted=True` and emits a
migration warning. Re-saving the object upgrades it to the current format.

Harness CLI and Studio only load bundles from the selected local workspace, and
must pass `trusted=True` explicitly. This acknowledges rather than eliminates the
trust boundary. Future remote artifact stores must establish provenance before
using that path.

## Compatibility

- Readers reject format versions newer than they support.
- Manifest inspection remains available even when payload dependencies are
  incompatible.
- Recorded dependency versions are diagnostic in version 1; strict package
  compatibility policies will be added after real cross-release evidence exists.
- Legacy support will not be removed without a separately documented migration.

## Validation

Tests must cover:

- save, inspect, trusted load, and prediction equality;
- refusal to deserialize without explicit trust;
- payload checksum corruption;
- unsupported format versions;
- missing archive members;
- atomic replacement behavior;
- trusted legacy loading and warning;
- training/config fingerprints and schemas in the manifest;
- CLI and Studio inference through the new loader;
- clean wheel-only train/save/inspect/load/predict in CI.
