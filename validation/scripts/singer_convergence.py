"""Run SINGER chains and diagnose agreement across four independent initial seeds."""
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 456, 789)
GLOBAL_METRICS = (
    "recombinations", "marginal_trees", "mean_pair_tmrca",
    "mean_total_branch_length", "diversity_fit_mse", "unmapped_mutations",
)


def file_hash(path):
    """Hash decompressed VCF bytes so gzip staging does not change identity."""
    digest = hashlib.sha256()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def expected_settings(args, seed, samples=None):
    samples = args.samples if samples is None else samples
    discarded = (args.burnin + args.thin - 1) // args.thin
    return dict(
        burnin_iterations_requested=args.burnin,
        burnin_iterations_effective=discarded * args.thin,
        burnin_samples=discarded, posterior_samples=samples,
        thin=args.thin, seed=seed, polar=0.99,
        first_posterior_index=discarded,
        end_posterior_index_exclusive=discarded + samples,
    )


def validate_run(path, settings, metadata, vcf_hash):
    if not (path / "SUCCESS").is_file():
        raise ValueError(f"Incomplete run at {path}. It will not be overwritten; use a new --output-root.")
    actual = read_json(path / "run.json")
    for key, value in settings.items():
        if actual.get(key) != value:
            raise ValueError(f"{path}: {key} is {actual.get(key)!r}, expected {value!r}")
    if read_json(path / "input_metadata.json") != metadata:
        raise ValueError(f"Input metadata differs in {path}")
    if file_hash(path / "input.vcf") != vcf_hash:
        raise ValueError(f"Input VCF differs in {path}")
    indices = range(settings["first_posterior_index"], settings["end_posterior_index_exclusive"])
    expected = {f"singer_{i}.trees" for i in indices}
    if {p.name for p in path.glob("singer_*.trees")} != expected:
        raise ValueError(f"Missing or unexpected posterior indices in {path}")


def reference_tools(path):
    """Use the same wrapper/converter as the reference, without executing logs."""
    commands = [shlex.split(line) for line in (path / "commands.txt").read_text().splitlines() if line.strip()]
    if len(commands) != 2 or any(len(command) < 2 for command in commands):
        raise ValueError("Reference commands.txt must contain the two run_singer.sh commands")
    master, converter = (Path(command[1]).resolve() for command in commands)
    if master.name != "singer_master" or not master.is_file() or not converter.is_file():
        raise ValueError("Cannot locate the reference singer_master and converter")
    return master, converter


def load_dependencies():
    # Reuse the optional supplements prepared in this session; no auto-install.
    supplement = os.environ.get("SINGER_DIAGNOSTICS_PATH")
    if supplement:
        sys.path.insert(0, str(Path(supplement).resolve()))
    elif importlib.util.find_spec("arviz") is None:
        cached = Path(tempfile.gettempdir()) / "singer_convergence_arviz"
        if not cached.is_dir():
            cached = Path("/tmp/singer_convergence_arviz")
        if cached.is_dir():
            sys.path.insert(0, str(cached))
            print(f"Using supplementary diagnostic packages from {cached}")
    # Keep plotting/JIT cache writes out of the conda environment.
    cache = Path(tempfile.gettempdir()) / "singer_convergence_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    try:
        import arviz as az
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import tskit
    except ImportError as exc:
        raise ValueError(
            "Diagnostic dependencies are unavailable. Install arviz==0.22.0, tskit, "
            "pandas and matplotlib in the active environment, or set SINGER_DIAGNOSTICS_PATH "
            "to a compatible supplemental package directory. " + str(exc)
        ) from exc
    return az, plt, np, pd, tskit


def extract_trace(path, settings, metadata, deps):
    _, _, np, pd, tskit = deps
    length = metadata["summary"]["sequence_length_bp"]
    samples = metadata["summary"]["num_haplotypes"]
    mu = metadata["simulation"]["sim_mutations"]["parameters"]["rate"]
    log = pd.read_csv(path / "raw/singer.log", sep="\t")
    log = log[log.Threading_type == "rethread"].set_index("Iteration:")
    if log.index.duplicated().any():
        raise ValueError(f"Duplicate saved-state indices in {path}/raw/singer.log")
    positions = [int(length * fraction) for fraction in (.1, .3, .5, .7, .9)]
    rows, digest = [], hashlib.sha256()
    for index in range(settings["first_posterior_index"], settings["end_posterior_index_exclusive"]):
        tree_path = path / f"singer_{index}.trees"
        digest.update(tree_path.name.encode())
        digest.update(tree_path.read_bytes())
        ts = tskit.load(tree_path)
        if ts.sequence_length != length or ts.num_samples != samples:
            raise ValueError(f"Unexpected sample count or sequence length in {tree_path}")
        branch_diversity = ts.diversity(mode="branch")
        row = dict(index=index, recombinations=log.loc[index, "#Recombinations"],
                   marginal_trees=ts.num_trees, mean_pair_tmrca=branch_diversity / 2,
                   mean_total_branch_length=sum(t.span * t.total_branch_length for t in ts.trees()) / length,
                   diversity_fit_mse=(ts.diversity(mode="site") - mu * branch_diversity) ** 2,
                   unmapped_mutations=log.loc[index, "#Mutations_not_uniquely_mapped"])
        for position in positions:
            tree = ts.at(position)
            if tree.num_roots != 1:
                raise ValueError(f"Multiple roots in {tree_path} at {position}")
            row[f"root_time_pos{position}"] = tree.time(tree.root)
            for a, b in itertools.combinations(ts.samples(), 2):
                row[f"tmrca_{a}_{b}_pos{position}"] = tree.tmrca(a, b)
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("index")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Nonfinite trace values in {path}")
    console = (path / "singer.log").read_text()
    provenance = dict(path=str(path), seed=settings["seed"], settings=settings,
                      posterior_sha256=digest.hexdigest(),
                      recoveries=console.count("Auto-debug iteration:"),
                      assertion_failures=console.count("Assertion"))
    return frame, provenance


def diagnose(traces, args, deps):
    az, _, np, pd, _ = deps
    rows = []
    for name in traces[0].columns:
        draws = np.stack([frame[name].to_numpy() for frame in traces])
        if np.ptp(draws) == 0:
            rows.append(dict(metric=name, status="constant_uninformative", rhat=np.nan,
                             ess_bulk=np.nan, ess_tail=np.nan, mcse_mean=np.nan,
                             mcse_over_sd=np.nan))
            continue
        row = dict(metric=name, rhat=float(az.rhat(draws, method="rank")),
                   ess_bulk=float(az.ess(draws, method="bulk")),
                   ess_tail=float(az.ess(draws, method="tail")),
                   mcse_mean=np.asarray(az.mcse(draws, method="mean")).item())
        row["mcse_over_sd"] = row["mcse_mean"] / draws.std(ddof=1)
        flags = []
        if not all(math.isfinite(value) for key, value in row.items() if key != "metric"):
            flags.append("undefined_diagnostic")
        if row["rhat"] >= args.max_rhat:
            flags.append("rhat")
        if min(row["ess_bulk"], row["ess_tail"]) < args.min_ess:
            flags.append("ess")
        if row["mcse_over_sd"] > args.max_mcse_sd:
            flags.append("mcse")
        row["status"] = ";".join(flags) if flags else "thresholds_met"
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def validate_baseline(args, vcf_hash):
    if args.baseline_report is None:
        return
    baseline = read_json(args.baseline_report / "manifest.json")
    if baseline["vcf_sha256"] != vcf_hash:
        raise ValueError("Baseline report is for a different VCF")
    chains = baseline["chains"]
    if sorted(c["seed"] for c in chains) != list(SEEDS):
        raise ValueError("Baseline report must contain exactly seeds 42, 123, 456, 789")
    counts = {c["settings"]["posterior_samples"] for c in chains}
    if len(counts) != 1 or next(iter(counts)) >= args.samples:
        raise ValueError("Baseline must have equal, shorter retained chains than this run")
    for chain in chains:
        expected = expected_settings(args, chain["seed"], samples=next(iter(counts)))
        if any(chain["settings"].get(key) != value for key, value in expected.items()):
            raise ValueError("Baseline burn-in, thinning, seeds and polarization must match")
        if not (args.baseline_report / f"seed{chain['seed']}_trace.csv").is_file():
            raise ValueError("Missing baseline trace CSV")
    if not (args.baseline_report / "diagnostics.csv").is_file():
        raise ValueError("Missing baseline diagnostics.csv")


def compare_baseline(traces, diagnostics, args, deps, report):
    """Descriptive comparison; old and longer reruns share initial seeds."""
    _, plt, np, pd, _ = deps
    old = pd.read_csv(args.baseline_report / "diagnostics.csv", index_col="metric")
    if set(old.index) != set(diagnostics.index):
        raise ValueError("Baseline monitored quantities differ from this report")
    old.join(diagnostics, lsuffix="_short", rsuffix="_long").to_csv(report / "short_vs_long_diagnostics.csv")
    late = [frame.iloc[len(frame) // 2:] for frame in traces]
    diagnose(late, args, deps).to_csv(report / "late_half_diagnostics.csv")
    rows = []
    baseline_manifest = read_json(args.baseline_report / "manifest.json")
    baseline_counts = {c["seed"]: c["settings"]["posterior_samples"] for c in baseline_manifest["chains"]}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    focus = ("mean_total_branch_length", "mean_pair_tmrca")
    for seed, frame in zip(SEEDS, traces):
        before = pd.read_csv(args.baseline_report / f"seed{seed}_trace.csv", index_col="index")
        if (not frame.columns.equals(before.columns)
                or len(before) != baseline_counts[seed]
                or not before.index.equals(frame.index[:len(before)])
                or not np.isfinite(before.to_numpy()).all()):
            raise ValueError(f"Invalid baseline trace for seed {seed}")
        for name in frame.columns:
            rows.append(dict(seed=seed, metric=name, short_mean=before[name].mean(),
                             long_mean=frame[name].mean(),
                             long_first_half_mean=frame[name].iloc[:len(frame)//2].mean(),
                             long_last_half_mean=frame[name].iloc[len(frame)//2:].mean()))
        window = min(50, max(1, len(frame) // 5))
        for ax, name in zip(axes, focus):
            line, = ax.plot(frame.index, frame[name].rolling(window).mean(), label=f"seed {seed}")
            ax.axhline(before[name].mean(), color=line.get_color(), ls="--", alpha=.6)
            ax.set_title(f"{name}: {window}-state moving means; dashed = short-run mean")
            ax.set_xlabel("Retained saved-state index")
            ax.legend()
    fig.tight_layout()
    fig.savefig(report / "stability.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(report / "short_vs_long_means.csv", index=False)
    return ["", "## Longer-run stability comparison", "",
            f"Baseline: `{args.baseline_report}`. The short and long runs share initial seeds; "
            "their estimates are correlated and are not treated as independent samples.", "",
            "[Short versus long diagnostics](short_vs_long_diagnostics.csv) · "
            "[Means by chain](short_vs_long_means.csv) · [Stability plot](stability.png)", "",
            "[Last-half diagnostics](late_half_diagnostics.csv) provide a secondary drift check. "
            "The main outcome still uses every retained draw after the pre-specified burn-in. "
            "A favorable last-half result does not override concerns in the full retained chain."]


def write_report(traces, provenance, diagnostics, args, deps, vcf_hash, master, converter):
    az, plt, np, pd, tskit = deps
    from matplotlib.backends.backend_pdf import PdfPages
    report = args.output_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(report / "diagnostics.csv")
    for frame, chain in zip(traces, provenance):
        frame.to_csv(report / f"seed{chain['seed']}_trace.csv")
    flagged = diagnostics[~diagnostics.status.isin(["thresholds_met", "constant_uninformative"])]
    constant = diagnostics[diagnostics.status == "constant_uninformative"]
    verdict = "diagnostic_concerns" if len(flagged) else "monitored_thresholds_met"
    summary = dict(verdict=verdict, convergence_proven=False, chains=len(traces),
                   draws_per_chain=args.samples, flagged_metrics=len(flagged),
                   uninformative_metrics=len(constant),
                   recovery_attempts=sum(chain["recoveries"] for chain in provenance),
                   thresholds=dict(rhat_below=args.max_rhat, ess_at_least=args.min_ess,
                                   mcse_over_sd_at_most=args.max_mcse_sd))
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                    arviz=az.__version__, arviz_path=az.__file__, tskit=tskit.__version__,
                    numpy=np.__version__, vcf_sha256=vcf_hash, dataset=str(args.dataset),
                    singer_master=str(master), converter=str(converter),
                    singer_master_sha256=file_hash(master), converter_sha256=file_hash(converter),
                    run_all_seeds=args.run_all_seeds,
                    baseline_report=str(args.baseline_report) if args.baseline_report else None,
                    chains=provenance)
    (report / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def plot_page(names):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        for ax, name in zip(axes.flat, names):
            for frame, chain in zip(traces, provenance):
                ax.plot(frame.index, frame[name], lw=.7, alpha=.75, label=f"seed {chain['seed']}")
            r = diagnostics.loc[name]
            ax.set_title(f"{name}\nR-hat {r.rhat:.3f}; bulk ESS {r.ess_bulk:.0f}; tail ESS {r.ess_tail:.0f}", fontsize=9)
            ax.set_xlabel("Retained saved-state index")
            if name.startswith(("tmrca_", "root_time_")):
                values = np.concatenate([frame[name].to_numpy() for frame in traces])
                if values.min() > 0 and values.max() / values.min() > 100:
                    ax.set_yscale("log")
                    ax.set_ylabel("Generations (log scale)")
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle(f"Same dataset, independent initial seeds; thin={args.thin}, burn-in={args.burnin}")
        fig.tight_layout()
        return fig

    names = list(diagnostics.index)
    with PdfPages(report / "all_traces.pdf") as pdf:
        for start in range(0, len(names), 6):
            fig = plot_page(names[start:start + 6])
            pdf.savefig(fig)
            plt.close(fig)
    worst = list(diagnostics.sort_values("rhat", ascending=False).index[:2])
    selected = list(dict.fromkeys(["recombinations", "mean_pair_tmrca", "diversity_fit_mse",
                                  "tmrca_0_5_pos2500", *worst]))
    selected = [name for name in selected if name in diagnostics.index][:6]
    fig = plot_page(selected)
    fig.savefig(report / "overview.png", dpi=150)
    plt.close(fig)
    lines = ["# SINGER independent-chain diagnostics", "",
             f"Outcome: **{verdict}**. {len(flagged)} monitored quantities have diagnostic concerns; "
             f"{len(constant)} constant quantities are uninformative. This is not proof of convergence.", "",
             f"Four chains on `{args.dataset}`: seeds 42, 123, 456, 789; {args.samples} retained states each, "
             f"burn-in {args.burnin}, thinning {args.thin}. " +
             ("All four full-length chains are in the output root; matching completed chains are reused. "
              f"The shorter reference `{args.reference_run}` supplies input/tool validation only."
              if args.run_all_seeds else f"Seed 42 was reused from `{args.reference_run}`."), "",
             "| Seed | Recovery attempts | Assertion failures |", "|---|---:|---:|"]
    lines.extend(f"| {c['seed']} | {c['recoveries']} | {c['assertion_failures']} |" for c in provenance)
    lines += ["", "[Diagnostic table](diagnostics.csv) · [Overview](overview.png) · [Every trace](all_traces.pdf)", "",
              f"Thresholds: rank-normalized split R-hat < {args.max_rhat}, bulk and tail ESS >= {args.min_ess}, "
              f"mean MCSE <= {args.max_mcse_sd:g} posterior SD. MCSE/SD is a screening rule, not a scientific accuracy guarantee.", "",
              "Global counts, genome-wide mean coalescence/branch length, and whole-window diversity-fit error are monitored, "
              "along with root ages and every pairwise TMRCA at 10%, 30%, 50%, 70%, and 90% of the sequence. "
              "Diversity-fit error is (site diversity - mutation rate × branch diversity) squared. "
              "Diagnostics use retained states only; no further burn-in is silently discarded.", "",
              "Constant pooled traces are not counted as evidence of mixing. A diagnostic failure or undefined R-hat "
              "for chains stuck at different values is flagged. Stable scalar summaries cannot establish exploration "
              "of the entire ARG distribution. Inspect trace plots for drift and time collapse.", "",
              "Recovery can roll back states and change seeds within a chain. Diagnostics do not establish that this "
              "recovery mechanism preserves the target distribution. Numerical failures require separate scrutiny.", "",
              "If diagnostics fail, investigate affected traces and sampler numerics before increasing run lengths. "
              "If thresholds are met, check scientific estimates across longer runs and their Monte Carlo errors before "
              "making a convergence claim. This script does not automatically launch longer chains.", "",
              "[R-hat, ESS and MCSE guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html) · "
              "[SINGER diagnostics](https://github.com/popgenmethods/SINGER#examining-the-convergence-of-the-mcmc-in-singer)", ""]
    if args.baseline_report:
        lines += compare_baseline(traces, diagnostics, args, deps, report)
    (report / "report.md").write_text("\n".join(lines))
    print(f"\n{verdict}: {len(flagged)} flagged quantities; {len(constant)} constant/uninformative.")
    print(f"Report: {report / 'report.md'}")


def run_workflow(args, runner=None):
    runner = runner or Path(__file__).with_name("run_singer.sh")
    metadata = read_json(args.dataset / "metadata.json")
    vcf = args.dataset / metadata["files"]["vcf"]
    vcf_hash = file_hash(vcf)
    paths = [args.output_root / "chains" / f"seed{seed}" for seed in SEEDS]
    if not args.run_all_seeds:
        paths[0] = args.reference_run
    else:
        if args.reference_run in paths:
            raise ValueError("The shorter reference must be outside the full-length chain directories")
        reference_samples = read_json(args.reference_run / "run.json")["posterior_samples"]
        validate_run(args.reference_run, expected_settings(args, 42, samples=reference_samples), metadata, vcf_hash)
    validate_baseline(args, vcf_hash)
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("Reference and new chains must have distinct directories")
    # Validate every existing run before starting any missing chain.
    for seed, path in zip(SEEDS, paths):
        if (seed == 42 and not args.run_all_seeds) or path.exists():
            validate_run(path, expected_settings(args, seed), metadata, vcf_hash)
    master, converter = reference_tools(args.reference_run)
    env = dict(os.environ, PYTHON_BIN=sys.executable, SINGER_BIN=str(master), CONVERT_TO_TSKIT=str(converter))
    deps = None if args.dry_run else load_dependencies()
    extracted = {}
    if deps:
        # Also load existing trees before launching expensive inference.
        for seed, path in zip(SEEDS, paths):
            if path.exists():
                extracted[seed] = extract_trace(path, expected_settings(args, seed), metadata, deps)
    for seed, path in zip(SEEDS, paths):
        if path.exists():
            print(f"Reuse completed seed {seed}: {path}")
            continue
        command = ["bash", str(runner), str(args.dataset), "--burnin", str(args.burnin),
                   "--samples", str(args.samples), "--thin", str(args.thin), "--seed", str(seed),
                   "--output-dir", str(path)]
        print(f"\nRun seed {seed}: {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, env=env, check=True)
        settings = expected_settings(args, seed)
        validate_run(path, settings, metadata, vcf_hash)
        extracted[seed] = extract_trace(path, settings, metadata, deps)
    if args.dry_run:
        print(f"\nWould compare all four chains and write {args.output_root / 'report'}")
        return
    traces, provenance = zip(*(extracted[seed] for seed in SEEDS))
    for frame in traces[1:]:
        if not frame.columns.equals(traces[0].columns) or not frame.index.equals(traces[0].index):
            raise ValueError("Chains do not have matching metric columns and posterior indices")
    diagnostics = diagnose(traces, args, deps)
    write_report(traces, provenance, diagnostics, args, deps, vcf_hash, master, converter)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "validation/datasets/human_25kb_super_easy/rep0")
    parser.add_argument("--reference-run", type=Path, help="Completed seed-42 reference for inputs and tools")
    parser.add_argument("--run-all-seeds", action="store_true", help="Run full-length seed 42 as well; reference supplies inputs/tools only")
    parser.add_argument("--baseline-report", type=Path, help="Compare longer chains with an existing four-chain report")
    parser.add_argument("--output-root", type=Path, help="New chains and report; existing completed chains are reused")
    parser.add_argument("--burnin", type=int, default=50000)
    parser.add_argument("--thin", type=int, default=500)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-rhat", type=float, default=1.01)
    parser.add_argument("--min-ess", type=float, default=400)
    parser.add_argument("--max-mcse-sd", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Validate run metadata/indices and print the plan; no writes or inference")
    args = parser.parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.reference_run = (args.reference_run or args.dataset.parent / "output" /
                          f"singer_thin{args.thin}_seed42" / args.dataset.name).resolve()
    args.output_root = (args.output_root or args.dataset.parent / "output" /
                       (f"singer_convergence_{args.dataset.name}" +
                        (f"_n{args.samples}" if args.run_all_seeds else ""))).resolve()
    if args.baseline_report:
        args.baseline_report = args.baseline_report.resolve()
        if args.baseline_report == args.output_root / "report":
            parser.error("Baseline report must not be the new report directory")
    if args.burnin < 0 or args.thin <= 0 or args.samples < 8:
        parser.error("burnin must be nonnegative, thin positive, and samples at least 8")
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_rhat, args.min_ess, args.max_mcse_sd)):
        parser.error("Diagnostic thresholds must be finite and positive")
    return args


if __name__ == "__main__":
    try:
        run_workflow(parse_args())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Error: {exc}")
