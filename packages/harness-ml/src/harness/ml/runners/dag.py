from harness.ml.config.models import SingleModelConfig


class ModelDAG:
    def __init__(self, models: dict[str, SingleModelConfig]):
        self._models = models
        self._deps = self._infer_dependencies()

    def _infer_dependencies(self) -> dict[str, set[str]]:
        """Build dependency map from depends_on declarations."""
        deps = {}
        for name, config in self._models.items():
            deps[name] = set(config.depends_on)
        return deps

    def topological_waves(self) -> list[list[str]]:
        """Kahn's algorithm — group into parallel waves.
        Wave 0: no dependencies. Wave N: depends only on waves 0..N-1.
        Returns list of lists of model names."""
        remaining = dict(self._deps)
        waves = []
        resolved = set()
        while remaining:
            # Find models with all deps resolved
            wave = [name for name, deps in remaining.items() if deps.issubset(resolved)]
            if not wave:
                # Cycle detected
                raise ValueError(f"Circular dependency among models: {list(remaining.keys())}")
            waves.append(sorted(wave))
            resolved.update(wave)
            for name in wave:
                del remaining[name]
        return waves

    def validate(self) -> list[str]:
        """Check for cycles, missing dependencies. Returns error messages."""
        errors = []
        all_names = set(self._models.keys())
        for name, deps in self._deps.items():
            for dep in deps:
                if dep not in all_names:
                    errors.append(f"Model '{name}' depends on '{dep}' which does not exist")
        # Check for cycles
        try:
            self.topological_waves()
        except ValueError as e:
            errors.append(str(e))
        return errors

    def dependencies(self, model_name: str) -> set[str]:
        return self._deps.get(model_name, set())
