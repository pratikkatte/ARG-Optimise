## tsinfer + tsdate on msprime VCF (run from scripts/; create .samples first with vcf_to_samples.py)
import os

import tskit
import tsinfer
import tsdate

simpref = "l1mb"  # sim_l1mb_0.{vcf,samples}; use "l25kb" for 25 kb
mu = 2e-8
rec = 2e-8
Ne = 10000
num_threads = 4

samples_path = f"../vcf/sim_{simpref}_0.samples"
inferred_path = f"../output/tsinfer/{simpref}.trees"
simpl_path = f"../output/tsinfer/{simpref}_simpl.trees"
dated_path = f"../output/tsinfer/{simpref}_dated.trees"

os.makedirs(os.path.dirname(inferred_path), exist_ok=True)
os.makedirs(os.path.dirname(dated_path), exist_ok=True)

print("loading", samples_path)
samples = tsinfer.SampleData.load(samples_path)

print("tsinfer.infer")
ts = tsinfer.infer(
    samples,
    recombination_rate=rec,
    mismatch_ratio=1.0,
    num_threads=num_threads,
    progress_monitor=True,
)
ts.dump(inferred_path)
print("wrote", inferred_path, ts.num_trees, "trees", ts.num_edges, "edges")

print("simplify")
tskit.load(inferred_path).simplify(keep_unary=False).dump(simpl_path)

print("tsdate.build_prior_grid")
priors = tsdate.build_prior_grid(
    tskit.load(simpl_path), population_size=Ne, timepoints=20
)

print("tsdate.date")
dated = tsdate.date(
    tskit.load(simpl_path),
    mutation_rate=mu,
    priors=priors,
    method="inside_outside",
)
dated.dump(dated_path)
print("wrote", dated_path, dated.num_trees, "trees", dated.num_edges, "edges")
