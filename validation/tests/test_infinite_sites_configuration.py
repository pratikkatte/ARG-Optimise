"""Integration tests for the full YAML-controlled infinite-sites workflow."""
import copy
import json
from pathlib import Path
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import patch
import numpy as np
import pytest
import torch
import yaml
from validation.tests.test_infinite_sites_neural import environment, model
from generator import GFlowNetGenerator
from gfn.rollout import RolloutWorker
from training.checkpoints import seed_everything, generator_from_checkpoint, restore_rng, rng_state, load_checkpoint
from training.configuration import resolve_config, parse_train_args, DEFAULTS
from training.schedules import LearningRateConfig, WarmupCosineScheduler, PolicyTemperatureConfig
from training.trainer import Trainer, TrajectoryMixConfig
from training.evaluation import evaluate_generator
from training.reporting import open_json
from train import train

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(autouse=True)
def deterministic():
    seed_everything(7); torch.set_num_threads(1)


def test_reference_keys_and_actionable_migration():
    old=yaml.safe_load((ROOT/'config/config_learned_event_2k_cosine_replay_fresh.yaml').read_text())
    with pytest.raises(ValueError,match='flow_head_version') as exc:
        resolve_config(old)
    assert 'breakpoint_dropout' in str(exc.value)
    old.update(flow_head_version=6,breakpoint_dropout=0.)
    c=resolve_config(old)
    assert c['epochs_num']==10000 and c['replay_capacity']==2048
    assert c['evaluation']['bank_per_stratum']==64
    with pytest.raises(ValueError,match='Unsupported'):
        resolve_config({**old,'typo_lr':1})
    new=resolve_config(parse_train_args(['--config',str(ROOT/'config/config_infinite_sites_rep0_cosine_replay.yaml'),
                                       '--no-wandb','--device','cpu','--epochs','2']))
    assert not new['wandb'] and new['device']=='cpu' and new['epochs_num']==2
    assert new['model_kwargs']['breakpoint_mixture_layers']==4


def test_architecture_time_and_breakpoint_controls():
    from env.actions import CoalescenceChoice
    from breakpoint_model import SparseMixtureBreakpointPolicy
    env=environment(recombination=.01,length=8)
    kwargs=dict(embedding_size=16,hidden_size=32,transformer_depth=2,transformer_heads=2,
                transformer_mlp_ratio=3,breakpoint_mixture_hidden_dim=19,breakpoint_mixture_layers=3,
                breakpoint_gap_hidden_size=11,breakpoint_gap_layers=2,breakpoint_mixture_components=3,
                continuous_time_head='exponential',time_hidden_dim=17,time_layers=1)
    g=GFlowNetGenerator(env,model_kwargs=kwargs,initialize_z_from_policy=False)
    assert g.state_encoder.transformer.blocks[0].mlp.fc1.out_features==48
    layers=[m for m in g.arg_model.breakpoint_head.parameters_head if isinstance(m,torch.nn.Linear)]
    assert [m.out_features for m in layers]==[19,19,19,11,11,9]
    assert g.arg_model.time_head.output_layer.in_features==17
    assert not hasattr(g.arg_model.time_head,'shape_layer')
    out=g([env.get_initial_state()],forced_actions=[CoalescenceChoice(1,2,delta_t=.2)])
    assert torch.isfinite(out['log_pf']).all()
    choice=env.enumerate_policy_actions(env.get_initial_state())[1][0]
    context=torch.zeros(64)
    scores=[g.arg_model.breakpoint_head(choice,context,8,breakpoint=i,temperature=2.)[1] for i in range(1,8)]
    assert float(torch.stack(scores).exp().sum().detach())==pytest.approx(1.,abs=1e-12)
    sampled,score=g.arg_model.breakpoint_head(choice,context,8,temperature=2.)
    torch.testing.assert_close(score,scores[sampled-1])


def test_tempered_proposal_and_current_policy_scores():
    env=environment(recombination=.01,length=4);g=model(env);w=RolloutWorker(env)
    with torch.no_grad():
        sampled,paths=w.rollout(g,2,random_spec={'T':2.,'time_T':1.})
        current,_=w.replay(g,paths)
    assert all(sum(p.log_proposals)==pytest.approx(float(sampled['log_paths_pf'][i].sum())) for i,p in enumerate(paths))
    assert not torch.equal(sampled['log_paths_pf'],current['log_paths_pf'])
    cfg=TrajectoryMixConfig(replay_fraction=0.)
    trainer=Trainer(g,w,cfg,temperature_config=PolicyTemperatureConfig('linear',2.,10))
    info=trainer.train_epoch(batch_size=2)
    assert info['policy_temperature']==2.
    with pytest.raises(ValueError,match='Temperature'):
        Trainer(g,w,TrajectoryMixConfig(),temperature_config=PolicyTemperatureConfig('linear',2.,10))


def test_accumulation_same_update_on_fixed_histories():
    env=environment();first=model(env);second=model(environment())
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        _,paths=RolloutWorker(env).rollout(first,6)
    def fixed_worker(g):
        w=RolloutWorker(g.env); cursor=[0]
        def rollout(generator,episodes,**kwargs):
            selected=paths[cursor[0]:cursor[0]+episodes];cursor[0]+=episodes
            return w.replay(generator, selected, collect_flows=kwargs.get('collect_flows', False),
                            return_states=kwargs.get('return_states', False))
        w.rollout=rollout
        return w
    a=Trainer(first,fixed_worker(first),TrajectoryMixConfig(replay_fraction=0.))
    b=Trainer(second,fixed_worker(second),TrajectoryMixConfig(replay_fraction=0.))
    one=a.train_epoch(batch_size=6,grad_accum_steps=1)
    three=b.train_epoch(batch_size=6,grad_accum_steps=3)
    assert one['loss']==pytest.approx(three['loss'],rel=1e-7,abs=1e-7)
    for x,y in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)


def test_cosine_resume_with_exploration_replay(tmp_path):
    g=model(environment(recombination=.001))
    schedule=LearningRateConfig('cosine',10,2,.2,.1)
    g.scheduler=WarmupCosineScheduler(g.opt,schedule)
    mix=TrajectoryMixConfig(exploration_fraction=.25,replay_fraction=.25,replay_capacity=10,
                            replay_grid_size=3,replay_per_topology=2,replay_min_size=2)
    t=Trainer(g,RolloutWorker(g.env),mix)
    t.train_epoch(batch_size=4,grad_accum_steps=2)
    g.save(tmp_path/'model.pt',trainer=t)
    checkpoint=load_checkpoint(tmp_path/'model.pt')
    expected=t.train_epoch(batch_size=4,grad_accum_steps=2)
    state=copy.deepcopy(g.state_dict())
    restored=generator_from_checkpoint(checkpoint,optimizer=True)
    t2=Trainer(restored,RolloutWorker(restored.env),mix);t2.load_state_dict(checkpoint['trainer'])
    restore_rng(restored.env,checkpoint['rng'])
    assert t2.train_epoch(batch_size=4,grad_accum_steps=2)==expected
    for name,value in restored.state_dict().items():
        torch.testing.assert_close(value,state[name],rtol=0,atol=0)
    assert schedule.factor(0)==.2 and schedule.factor(2)==1. and schedule.factor(10)==.1


def test_evaluation_isolates_rng_and_does_not_change_next_update():
    g=model();g.initialize_flow_center();t=Trainer(g,RolloutWorker(g.env),TrajectoryMixConfig(replay_fraction=0.))
    state=rng_state(g.env);before=copy.deepcopy(g.state_dict())
    with patch('training.evaluation.validate_terminal',side_effect=AssertionError('disabled')):
        metrics,details=evaluate_generator(g,4,independent=False)
    assert metrics['eval_independent_checked']==0 and len(details['records'])==4
    assert g.training
    after=rng_state(g.env)
    assert torch.equal(after['torch'],state['torch']) and after['environment']==state['environment']
    for name,value in g.state_dict().items(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    with patch('env.env.evaluate_infinite_sites',side_effect=AssertionError('training called reference')):
        t.train_epoch(batch_size=2)
    metrics,_=evaluate_generator(g,4,independent=True)
    assert metrics['eval_independent_checked']==4 and metrics['eval_max_likelihood_error']<1e-9


def fixture_dataset(tmp_path):
    path=tmp_path/'observed';path.mkdir()
    (path/'metadata.json').write_text(json.dumps(dict(parameters=dict(population_size=10,
                                            mutation_rate=.025,recombination_rate=0.))))
    return path,environment().snp_data


def test_full_cli_training_eval_wandb_and_resume(tmp_path, capsys):
    path,data=fixture_dataset(tmp_path)
    options=dict(dataset_path=str(path),output_path=str(tmp_path/'run'),epochs=2,batch_size=4,
        model_kwargs=dict(embedding_size=16,hidden_size=32,transformer_depth=1,transformer_heads=2),
        init_z_sample_count=2,init_z_batch_size=1,eval_episodes=4,eval_every=2,checkpoint_every=1,grad_accum_steps=2,
        replay_capacity=8,replay_grid_size=3,replay_per_topology=1,replay_min_size=2,
        lr_schedule='cosine',lr_schedule_steps=4,wandb=True,wandb_mode='offline',verbose=True)
    class Run:
        id='test-run'
        def __init__(self): self.logged=[];self.finished=False;self.summary={}
        def log(self,info,step): self.logged.append((step,info))
        def finish(self): self.finished=True
    run=Run()
    wandb=ModuleType('wandb')
    wandb.__spec__=ModuleSpec('wandb',loader=None)
    wandb.init=lambda **kwargs:run
    with patch('train.load_snp_dataset',return_value=data),patch.dict(
            'sys.modules', wandb=wandb):
        g,t=train(**options)
    lines=capsys.readouterr().out.splitlines()
    initialization=[line for line in lines if line.startswith(('Initializing flow', 'Initialization complete'))]
    assert len(initialization)==2
    assert initialization[0]=='Initializing flow (2 trajectories)...'
    assert initialization[1].startswith('Initialization complete (') and initialization[1].endswith('s)')
    batches=[line for line in lines if line.startswith('Z init ')]
    assert len(batches)==2
    for index,line in enumerate(batches,1):
        assert line.startswith(f'Z init {index}/2 | events/ARG=') and line.endswith('s')
    epochs=[line for line in lines if line.startswith('Epoch ')]
    assert len(epochs)==2
    assert epochs[0].startswith('Epoch 1/2  subtb_loss=') and 'eval_subtb_loss=' not in epochs[0]
    assert epochs[1].startswith('Epoch 2/2  subtb_loss=') and 'eval_subtb_loss=' in epochs[1]
    assert all('  time=' in line for line in epochs)
    assert run.finished and len(run.logged)==2
    assert 'grad_norm' in run.logged[0][1] and 'eval_subtb_loss' in run.logged[1][1]
    training=[json.loads(s) for s in (tmp_path/'run/training.jsonl').read_text().splitlines()]
    for line,info in zip(epochs,training):
        assert f'subtb_loss={info["loss"]:.4f}' in line
        assert 'grad_norm' in info
    assert lines==initialization[:1]+batches+initialization[1:]+epochs and 'progress' not in run.summary
    assert not (tmp_path/'run/progress.jsonl').exists()
    resolved=yaml.safe_load((tmp_path/'run/resolved_config.yaml').read_text())
    assert resolved['effective_population_size']==10 and resolved['lr_schedule_steps']==4
    assert t.buffer.grid_size==3 and t.buffer.per_topology==1
    assert (tmp_path/'run/checkpoints/best_eval.pt').exists()
    reports=[json.loads(s) for s in (tmp_path/'run/evaluation.jsonl').read_text().splitlines()]
    assert len(reports)==1 and reports[0]['eval_independent_checked']==4
    assert f'eval_subtb_loss={reports[0]["eval_subtb_loss"]:.4f}' in epochs[1]
    checkpoint=tmp_path/'run/checkpoints/checkpoint_0002.pt'
    # Older checkpoints may contain the retired logging settings.
    saved=load_checkpoint(checkpoint)
    saved['metadata']['resolved_config'].update(debug_progress=True,progress_every_seconds=15.)
    torch.save(saved,checkpoint)
    # No original observation directory is needed for resume without truth evaluation.
    with patch('train.load_snp_dataset',side_effect=AssertionError('dataset should be embedded')):
        resumed,trainer=train(output_path=str(tmp_path/'resume'),resume_checkpoint=str(checkpoint),
                              epochs=3,wandb=False,verbose=False)
    assert trainer.completed_updates==3 and resumed.scheduler.completed_updates==3
    assert capsys.readouterr().out==''
    assert not (tmp_path/'resume/progress.jsonl').exists()
    resumed_config=yaml.safe_load((tmp_path/'resume/resolved_config.yaml').read_text())
    assert 'debug_progress' not in resumed_config and 'progress_every_seconds' not in resumed_config


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable')
def test_cuda_configured_training():
    env=environment(recombination=.001)
    g=GFlowNetGenerator(env,device='cuda',initialize_z_from_policy=False,
        model_kwargs=dict(embedding_size=16,hidden_size=32,transformer_depth=1,transformer_heads=2))
    t=Trainer(g,RolloutWorker(env),TrajectoryMixConfig(replay_fraction=0.))
    assert np.isfinite(t.train_epoch(batch_size=2)['loss'])
    assert next(g.parameters()).is_cuda and env.device=='cpu'


def test_standalone_evaluation_mapping_density_bank_and_repeats(tmp_path):
    from eval.eval import parse_eval_args, run_evaluation
    g=model();g.initialize_flow_center()
    g.save(tmp_path/'model.pt')
    config=dict(output_path=str(tmp_path/'run'),evaluation=dict(checkpoint=str(tmp_path/'model.pt'),
        metrics=['density_fit','ess'],num_samples=4,repeats=2,batch_size=2,seed=100007,device='cpu',
        grid_size=3,rank_bins=7,bank_per_stratum=1,bank_candidates=12))
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    options=parse_eval_args(['--config',str(tmp_path/'config.yaml'),'--output-dir',str(tmp_path/'eval')])
    result=run_evaluation(options)
    assert result['num_completed']==8 and len(result['repeats'])==2
    assert result['density_bank']['count']==3 and result['density_bank']['candidate_count']==12
    bank=json.loads((tmp_path/'eval/density_bank.json').read_text())
    assert bank['importance_ess'] is None and len(bank['all_candidates'])==12
    assert all(row['provenance']['source']=='compatible_proposal' for row in bank['records'])


def test_truth_evaluation_repeats_and_rank_bins(tmp_path):
    from eval.posterior_summary import TerminalSamplingEvaluator
    import tskit
    env=environment()
    tables=tskit.TableCollection(env.sequence_length);tables.time_units='generations'
    for _ in range(3): tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0)
    tables.nodes.add_row(time=10);tables.nodes.add_row(time=20)
    for parent,child in ((3,1),(3,2),(4,0),(4,3)):
        tables.edges.add_row(0,env.sequence_length,parent,child)
    tables.sort()
    evaluator=TerminalSamplingEvaluator(tables.tree_sequence(),[0,1,2],env.snp_data.haplotype_ids,
                    env.population_size,grid_size=3,tmrca_method='point_accuracy',rank_bins=7)
    metrics,details=evaluate_generator(model(env),4,terminal_evaluator=evaluator)
    assert details['truth_protocol']['calibration_definition']['rank_bins']==7
    assert metrics['eval_truth_pair_tmrca_rmse']>=0
    path,data=fixture_dataset(tmp_path)
    with patch('train.load_snp_dataset',return_value=data),patch(
            'eval.posterior_summary.TerminalSamplingEvaluator.from_dataset',return_value=evaluator) as factory:
        train(dataset_path=str(path),output_path=str(tmp_path/'truth_run'),epochs=2,batch_size=2,
              init_z_sample_count=2,model_kwargs=dict(embedding_size=16,hidden_size=32,
              transformer_depth=1,transformer_heads=2),eval_episodes=4,eval_every=1,terminal_eval=True,
              terminal_eval_grid_size=3,terminal_eval_repeats=3,terminal_eval_repeat_every=2,
              verbose=False,replay_fraction=0.)
    assert factory.call_count==2
    rows=[json.loads(s) for s in (tmp_path/'truth_run/evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==4 and [r['repeat'] for r in rows]==[0,0,1,2]
    reports=sorted((tmp_path/'truth_run/evaluation').glob('*.json.gz'))
    assert len(reports)==4
    for row,path in zip(rows,reports):
        with open_json(path) as handle:
            report=json.load(handle)
        assert report['metrics']=={k:v for k,v in row.items() if k not in ('step','repeat')}
        assert 'pair_tmrca_exact' in report['details']['truth']
