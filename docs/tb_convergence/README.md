# TB convergence and importance ESS

This directory tracks the diagnosis and remediation of posterior-quality
failures in the ARG GFlowNet.

- [issues.md](issues.md) records all currently known accuracy and training
  problems, including issues intentionally deferred from this phase.
- [fix-01-importance-ess-diagnostics.md](fix-01-importance-ess-diagnostics.md)
  documents trajectory-balance and importance-weight diagnostics.
- [fix-02-training-stabilization.md](fix-02-training-stabilization.md) documents
  reward, logZ, gradient, and training configuration changes.
- [fix-03-checkpoint-and-inference-gates.md](fix-03-checkpoint-and-inference-gates.md)
  documents checkpoint selection and safe inference behavior.

The code changes are complete and covered by automated tests. A full POC
retraining run is still required before any new checkpoint can be described as
converged. Passing the gate establishes proposal quality; it does not by itself
resolve the deferred topology, time-bin, encoder, or breakpoint problems.

## POC workflow

```bash
python train.py --config config_poc2.yaml
python infer.py \
  --checkpoint outputs/example2_tb_stable/checkpoints/best.pt \
  --output-dir outputs/example2_tb_stable/inference_heldout \
  --num-args 256 --batch-size 16 --seed 900007
```

If training never creates `best.pt`, inspect `convergence_report.json` and
`best_candidate.pt`. Diagnostic sampling is explicit:

```bash
python infer.py \
  --checkpoint outputs/example2_tb_stable/checkpoints/best_candidate.pt \
  --output-dir outputs/example2_tb_stable/diagnostic_samples \
  --num-args 32 --allow-unconverged
```

