"""Strict YAML configuration for the validation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import OUTPUT_ROOT, experiment_gfn_dir, validate_experiment_name


METHOD_NAMES = ("gfn", "tsinfer", "singer")
ROOT_KEYS = {"truth", "methods", "validation"}
TRUTH_KEYS = {"trees", "tracks_dir", "tracks_prefix", "ne", "haplotypes"}
VALIDATION_KEYS = {
    "skip",
    "max_pairs",
    "pair_seed",
    "xlim",
    "ylim",
    "xlim_log",
    "ylim_log",
    "verbose",
    "region",
}
METHOD_KEYS = {
    "label",
    "inferred_trees",
    "input_dir",
    "sample_prefix",
    "bed_dir",
    "bed_prefix",
    "mcspl",
    "burnin_samples",
    "max_posterior_samples",
}


class ConfigurationError(ValueError):
    """Raised when validation configuration is invalid."""


@dataclass(frozen=True)
class TruthConfig:
    trees: Path | None
    tracks_dir: Path | None
    tracks_prefix: str | None
    ne: float
    haplotypes: int


@dataclass(frozen=True)
class MethodConfig:
    name: str
    mode: str
    label: str | None = None
    inferred_trees: Path | None = None
    input_dir: Path | None = None
    sample_prefix: str | None = None
    bed_dir: Path | None = None
    bed_prefix: str | None = None
    mcspl: str | None = None
    burnin_samples: int = 0
    max_posterior_samples: int | None = None


@dataclass(frozen=True)
class ValidationOptions:
    skip: int = 1
    max_pairs: int | None = None
    pair_seed: int = 42
    xlim: str = "0,16"
    ylim: str = "0,8"
    xlim_log: str = "-4,1.5"
    ylim_log: str = "-4,1.5"
    verbose: bool = False
    region: tuple[float, float] | None = None


@dataclass(frozen=True)
class PipelineConfig:
    source: Path
    experiment: str
    output_root: Path
    truth: TruthConfig
    methods: dict[str, MethodConfig]
    validation: ValidationOptions


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(
            f"{location} contains unknown key(s): {', '.join(unknown)}"
        )


def _required(data: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in data:
        raise ConfigurationError(f"{location}.{key} is required")
    return data[key]


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{location} must be at least {minimum}")
    return value


def _number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{location} must be a number")
    result = float(value)
    if minimum is not None and result <= minimum:
        raise ConfigurationError(f"{location} must be greater than {minimum}")
    return result


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be true or false")
    return value


def _path(value: Any, location: str, base_dir: Path) -> Path:
    text = _string(value, location)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(
    data: Mapping[str, Any], key: str, location: str, base_dir: Path
) -> Path | None:
    value = data.get(key)
    if value is None:
        return None
    return _path(value, f"{location}.{key}", base_dir)


def _optional_string(
    data: Mapping[str, Any], key: str, location: str
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    return _string(value, f"{location}.{key}")


def _limit(value: Any, location: str, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ConfigurationError(
            f"{location} must be 'LOW,HIGH' or a two-number list"
        )
    if len(parts) != 2:
        raise ConfigurationError(f"{location} must contain exactly two values")
    try:
        low, high = (float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{location} values must be numbers") from exc
    if low >= high:
        raise ConfigurationError(f"{location} lower bound must be below upper bound")
    return f"{low:g},{high:g}"


def _parse_truth(data: Any, base_dir: Path) -> TruthConfig:
    truth = _mapping(data, "truth")
    _reject_unknown(truth, TRUTH_KEYS, "truth")
    trees = _optional_path(truth, "trees", "truth", base_dir)
    tracks_dir = _optional_path(truth, "tracks_dir", "truth", base_dir)
    tracks_prefix = _optional_string(truth, "tracks_prefix", "truth")
    has_trees = trees is not None
    has_tracks = tracks_dir is not None or tracks_prefix is not None
    if has_trees == has_tracks:
        raise ConfigurationError(
            "truth must define exactly one source: trees, or tracks_dir plus "
            "tracks_prefix"
        )
    if has_tracks and (tracks_dir is None or tracks_prefix is None):
        raise ConfigurationError(
            "truth.tracks_dir and truth.tracks_prefix must be provided together"
        )
    ne = _number(_required(truth, "ne", "truth"), "truth.ne", minimum=0)
    haplotypes = _integer(
        _required(truth, "haplotypes", "truth"),
        "truth.haplotypes",
        minimum=2,
    )
    return TruthConfig(
        trees=trees,
        tracks_dir=tracks_dir,
        tracks_prefix=tracks_prefix,
        ne=ne,
        haplotypes=haplotypes,
    )


def _parse_method(
    name: str,
    data: Any,
    *,
    base_dir: Path,
    experiment: str,
    output_root: Path,
) -> MethodConfig:
    location = f"methods.{name}"
    method = _mapping(data, location)
    _reject_unknown(method, METHOD_KEYS, location)

    label = _optional_string(method, "label", location)
    inferred_trees = _optional_path(method, "inferred_trees", location, base_dir)
    input_dir = _optional_path(method, "input_dir", location, base_dir)
    sample_prefix = _optional_string(method, "sample_prefix", location)
    bed_dir = _optional_path(method, "bed_dir", location, base_dir)
    bed_prefix = _optional_string(method, "bed_prefix", location)
    mcspl = _optional_string(method, "mcspl", location)

    burnin_samples = _integer(
        method.get("burnin_samples", 0),
        f"{location}.burnin_samples",
        minimum=0,
    )
    max_samples_value = method.get("max_posterior_samples")
    max_posterior_samples = (
        None
        if max_samples_value is None
        else _integer(
            max_samples_value,
            f"{location}.max_posterior_samples",
            minimum=1,
        )
    )

    if name == "gfn" and all(
        value is None
        for value in (inferred_trees, input_dir, bed_dir, bed_prefix, mcspl)
    ):
        input_dir = experiment_gfn_dir(experiment, output_root=output_root)
        sample_prefix = sample_prefix or "arg_"

    has_single = inferred_trees is not None
    has_samples = input_dir is not None or sample_prefix is not None
    has_bed = bed_dir is not None or bed_prefix is not None or mcspl is not None
    mode_count = sum((has_single, has_samples, has_bed))
    if mode_count != 1:
        raise ConfigurationError(
            f"{location} must define exactly one input mode: inferred_trees, "
            "input_dir plus sample_prefix, or BED fields"
        )

    if has_samples and (input_dir is None or sample_prefix is None):
        raise ConfigurationError(
            f"{location}.input_dir and {location}.sample_prefix must be provided together"
        )
    if has_bed:
        if name == "gfn":
            raise ConfigurationError("methods.gfn does not support BED input")
        if bed_dir is None or bed_prefix is None:
            raise ConfigurationError(
                f"{location}.bed_dir and {location}.bed_prefix must be provided together"
            )
        if name == "singer" and mcspl is None:
            raise ConfigurationError(f"{location}.mcspl is required for BED input")
        if name == "tsinfer" and mcspl is not None:
            raise ConfigurationError(f"{location}.mcspl is only valid for SINGER")
    if name == "tsinfer" and has_samples:
        raise ConfigurationError("methods.tsinfer does not support posterior samples")
    if name != "singer" and mcspl is not None:
        raise ConfigurationError(f"{location}.mcspl is only valid for SINGER")
    if not has_samples and (
        burnin_samples != 0 or max_posterior_samples is not None
    ):
        raise ConfigurationError(
            f"{location} burnin/maximum settings require posterior samples"
        )

    mode = "trees" if has_single else "samples" if has_samples else "bed"
    return MethodConfig(
        name=name,
        mode=mode,
        label=label,
        inferred_trees=inferred_trees,
        input_dir=input_dir,
        sample_prefix=sample_prefix,
        bed_dir=bed_dir,
        bed_prefix=bed_prefix,
        mcspl=mcspl,
        burnin_samples=burnin_samples,
        max_posterior_samples=max_posterior_samples,
    )


def _parse_validation(data: Any) -> ValidationOptions:
    if data is None:
        validation: Mapping[str, Any] = {}
    else:
        validation = _mapping(data, "validation")
    _reject_unknown(validation, VALIDATION_KEYS, "validation")

    max_pairs_value = validation.get("max_pairs")
    max_pairs = (
        None
        if max_pairs_value is None
        else _integer(max_pairs_value, "validation.max_pairs", minimum=1)
    )
    region_value = validation.get("region")
    region: tuple[float, float] | None = None
    if region_value is not None:
        if not isinstance(region_value, (list, tuple)) or len(region_value) != 2:
            raise ConfigurationError(
                "validation.region must be a two-number [left, right] interval"
            )
        left = _number(region_value[0], "validation.region[0]")
        right = _number(region_value[1], "validation.region[1]")
        if left < 0 or right <= left:
            raise ConfigurationError(
                "validation.region must satisfy 0 <= left < right"
            )
        region = (left, right)
    return ValidationOptions(
        skip=_integer(validation.get("skip", 1), "validation.skip", minimum=1),
        max_pairs=max_pairs,
        pair_seed=_integer(
            validation.get("pair_seed", 42), "validation.pair_seed"
        ),
        xlim=_limit(validation.get("xlim"), "validation.xlim", "0,16"),
        ylim=_limit(validation.get("ylim"), "validation.ylim", "0,8"),
        xlim_log=_limit(
            validation.get("xlim_log"), "validation.xlim_log", "-4,1.5"
        ),
        ylim_log=_limit(
            validation.get("ylim_log"), "validation.ylim_log", "-4,1.5"
        ),
        verbose=_boolean(
            validation.get("verbose", False), "validation.verbose"
        ),
        region=region,
    )


def load_config(
    path: Path | str,
    experiment: str,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> PipelineConfig:
    """Load and structurally validate a validation YAML file."""
    experiment = validate_experiment_name(experiment)
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"validation config does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"could not parse validation YAML: {exc}") from exc
    root = _mapping(raw, "config")
    _reject_unknown(root, ROOT_KEYS, "config")
    truth = _parse_truth(_required(root, "truth", "config"), source.parent)
    methods_raw = _mapping(_required(root, "methods", "config"), "methods")
    unknown_methods = sorted(set(methods_raw) - set(METHOD_NAMES))
    if unknown_methods:
        raise ConfigurationError(
            f"methods contains unknown method(s): {', '.join(unknown_methods)}"
        )
    if not methods_raw:
        raise ConfigurationError("methods must configure at least one method")
    configured_method_names = [
        name for name in METHOD_NAMES if name in methods_raw
    ]
    methods = {
        name: _parse_method(
            name,
            methods_raw[name],
            base_dir=source.parent,
            experiment=experiment,
            output_root=Path(output_root).resolve(),
        )
        for name in configured_method_names
    }
    return PipelineConfig(
        source=source,
        experiment=experiment,
        output_root=Path(output_root).resolve(),
        truth=truth,
        methods=methods,
        validation=_parse_validation(root.get("validation")),
    )


def _require_file(path: Path | None, label: str) -> None:
    if path is None or not path.is_file():
        raise ConfigurationError(f"{label} does not exist: {path}")


def _require_directory(path: Path | None, label: str) -> None:
    if path is None or not path.is_dir():
        raise ConfigurationError(f"{label} does not exist: {path}")


def _pairs(haplotypes: int, skip: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(0, haplotypes - 1, skip)
        for right in range(left + 1, haplotypes, skip)
    ]


def _preflight_bed(
    method: MethodConfig, *, haplotypes: int, skip: int
) -> None:
    assert method.bed_dir is not None
    assert method.bed_prefix is not None
    missing: list[str] = []
    for left, right in _pairs(haplotypes, skip):
        pair = f"{left}-{right}"
        if method.name == "tsinfer":
            candidates = (
                method.bed_dir / f"{method.bed_prefix}{pair}_post.bed.gz",
                method.bed_dir / f"{method.bed_prefix}{pair}_post.bed",
            )
        else:
            candidates = (
                method.bed_dir
                / (
                    f"arg-sample_sim_{method.bed_prefix}_{pair}_Tcpair_"
                    f"spl{method.mcspl}_posterior.bed"
                ),
                method.bed_dir / f"{method.bed_prefix}_pair_{pair}_posterior.bed",
            )
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(pair)
    if missing:
        preview = ", ".join(missing[:5])
        remainder = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ConfigurationError(
            f"methods.{method.name} is missing BED input for pair(s): "
            f"{preview}{remainder}"
        )


def preflight_config(config: PipelineConfig) -> None:
    """Check that every configured input needed by all methods exists."""
    truth = config.truth
    if truth.trees is not None:
        _require_file(truth.trees, "truth.trees")
    else:
        _require_directory(truth.tracks_dir, "truth.tracks_dir")
        assert truth.tracks_dir is not None
        assert truth.tracks_prefix is not None
        missing_truth = [
            f"{left}-{right}"
            for left, right in _pairs(
                truth.haplotypes, config.validation.skip
            )
            if not (
                truth.tracks_dir
                / f"{truth.tracks_prefix}_spls{left}-{right}.tc"
            ).is_file()
        ]
        if missing_truth:
            preview = ", ".join(missing_truth[:5])
            remainder = (
                ""
                if len(missing_truth) <= 5
                else f" (+{len(missing_truth) - 5} more)"
            )
            raise ConfigurationError(
                f"truth is missing track files for pair(s): {preview}{remainder}"
            )

    for method in config.methods.values():
        if method.mode == "trees":
            _require_file(
                method.inferred_trees, f"methods.{method.name}.inferred_trees"
            )
        elif method.mode == "samples":
            _require_directory(
                method.input_dir, f"methods.{method.name}.input_dir"
            )
            assert method.input_dir is not None
            assert method.sample_prefix is not None
            samples = [
                path
                for path in method.input_dir.glob(
                    f"{method.sample_prefix}*.trees"
                )
                if path.is_file()
            ]
            if not samples:
                raise ConfigurationError(
                    f"methods.{method.name} found no "
                    f"{method.sample_prefix}*.trees files in {method.input_dir}"
                )
            if method.burnin_samples >= len(samples):
                raise ConfigurationError(
                    f"methods.{method.name}.burnin_samples drops all "
                    f"{len(samples)} available samples"
                )
        else:
            _require_directory(
                method.bed_dir, f"methods.{method.name}.bed_dir"
            )
            _preflight_bed(
                method,
                haplotypes=truth.haplotypes,
                skip=config.validation.skip,
            )
