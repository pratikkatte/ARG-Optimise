"""Isolated, fresh-policy evaluation; never used to construct training rewards."""
from contextlib import contextmanager
import numpy as np
import torch
from training.checkpoints import rng_state, restore_rng, seed_everything
from gfn.rollout import RolloutWorker
from eval.density_fit import density_summary
from eval.ess import importance_stats
from infer import validate_terminal


@contextmanager
def preserve_sampling(generator):
    state = rng_state(generator.env)
    modes = [(m,m.training) for m in generator.modules()]
    try:
        generator.eval()
        yield
    finally:
        restore_rng(generator.env,state)
        for m,mode in modes:
            m.training = mode


@torch.no_grad()
def evaluate_generator(generator, episodes, batch_size=2, seed=100007, max_events=10000,
                       density=True, independent=True, terminal_evaluator=None):
    if episodes < 1 or batch_size < 1:
        raise ValueError('Evaluation counts must be positive')
    records, trees, loss_sum = [], [], 0.
    progress = getattr(generator, 'progress_reporter', None)
    if progress is not None:
        progress.begin('evaluation_sampling', evaluation_completed=0, evaluation_total=episodes, seed=seed)
    with preserve_sampling(generator):
        seed_everything(seed); generator.env.rng.seed(seed)
        worker = RolloutWorker(generator.env,max_events=max_events)
        for start in range(0,episodes,batch_size):
            count = min(batch_size,episodes-start)
            outputs, paths = worker.rollout(generator,count,collect_flows=True,return_states=True)
            loss_sum += float(generator.get_loss_from_rollout_outputs(outputs))*count
            if progress is not None:
                progress.update(force=True, activity='independent_validation' if independent else 'collecting_scores',
                                evaluation_completed=start, evaluation_total=episodes,
                                batch_completed=count, batch_total=count,
                                events_max=max(len(path) for path in paths), active_lineages_max=0)
            for i,state in enumerate(outputs['states']):
                reference = validate_terminal(generator.env,state) if independent else None
                pf = float(outputs['log_paths_pf'][i].sum())
                records.append(dict(source='fresh_untempered_policy',log_reward=state.log_reward,
                    log_likelihood=state.partial_log_likelihood,log_prior=state.accumulated_log_prior,
                    log_policy_density=pf,log_backward_probability=0.,event_count=len(paths[i]),
                    recombinations=sum(a.event_type=='recomb' for a in paths[i].actions),
                    likelihood_error=abs(reference.log_likelihood-state.partial_log_likelihood) if reference else None))
                if terminal_evaluator is not None:
                    trees.append(generator.env.save_to_tree_sequence(state))
                if progress is not None:
                    progress.update(evaluation_completed=len(records))
            if progress is not None:
                progress.update(force=True, evaluation_completed=len(records), activity='sampling')
        importance = importance_stats([r['log_reward']-r['log_policy_density'] for r in records],
                                     reward_constant=generator.env.reward_fn.C)
        metrics = dict(eval_subtb_loss=loss_sum/episodes,
                    eval_source='fresh_untempered_policy',
                    **{'eval_'+k:v for k,v in importance.items()},
                    eval_mean_events=float(np.mean([r['event_count'] for r in records])),
                    eval_max_events=max(r['event_count'] for r in records),
                    eval_independent_checked=episodes if independent else 0,
                    eval_max_likelihood_error=max(r['likelihood_error'] for r in records) if independent else None)
        details = dict(records=records)
        if density:
            fit = density_summary(records)
            details['density_fit'] = fit
            for key in ('raw','prior_relative'):
                metrics.update({'eval_density_'+key+'_'+name:value
                    for name,value in fit[key]['global_fit'].items() if name in ('slope','pearson','rmse')})
        if terminal_evaluator is not None:
            if progress is not None:
                progress.begin('evaluation_truth_summary', trajectories=episodes)
            summary, truth_details = terminal_evaluator.summarize_trees(trees)
            metrics.update(summary); details['truth'] = truth_details
            details['truth_protocol'] = terminal_evaluator.protocol
    if progress is not None:
        progress.begin('evaluation_complete', trajectories=episodes, loss=metrics['eval_subtb_loss'])
    return metrics,details
