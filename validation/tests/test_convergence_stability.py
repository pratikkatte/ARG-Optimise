"""Numerical failure witnesses, honest run status, and checkpoint semantics."""
import json
import math
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from env.actions import CoalescenceChoice
from env.env import SimpleARGEnvironment
from env.infinite_sites import evaluate_infinite_sites
from env.snp_data import SNPData
from infer import validate_terminal, TerminalValidationError
from policy.time_model import CwrGammaMixtureTimeModel
from training.checkpoints import load_checkpoint, generator_from_checkpoint
from validation.tests.test_infinite_sites_neural import model, environment
from validation.tests.test_infinite_sites_configuration import fixture_dataset
from gfn.rollout import RolloutWorker
from train import train


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_short_branch_roundoff_and_real_mismatch():
    data=SNPData(np.array([[1],[1],[0]],dtype=np.uint8),np.array([.5]),1.,
                 (0,),('A',),('G',),('a','b','c'))
    env=SimpleARGEnvironment(snp_data=data,population_size=10000,mutation_rate=1e-8,recombination_rate=0)
    state=env.apply_action(env.get_initial_state(),CoalescenceChoice(0,1,delta_t=12.7))
    state=env.apply_action(state,CoalescenceChoice(0,1,delta_t=1e-10))
    exported=evaluate_infinite_sites(env.save_to_tree_sequence(state),data,mutation_rate=env.mutation_rate)
    assert abs(exported.log_likelihood-state.partial_log_likelihood)>1e-9
    stable=validate_terminal(env,state)
    assert abs(stable.log_likelihood-state.partial_log_likelihood)<1e-12
    state.partial_log_likelihood+=.01
    with pytest.raises(TerminalValidationError) as error:
        validate_terminal(env,state)
    assert len(error.value.details['actions'])==2
    assert abs(error.value.details['log_likelihood']-error.value.details['reference_log_likelihood'])>.009


def test_bounded_mixture_extreme_outputs_and_exact_density():
    head=CwrGammaMixtureTimeModel(2,8,0.,components=2,parameterization='bounded_v1')
    extreme=torch.tensor([[1405.,-1435.,-735.,-634.,-932.,1187.]],dtype=torch.float64)
    logw,shape,lograte,rate=head.mixture_parameters(extreme,torch.tensor([3.]))
    assert bool((shape>=1).all() and (shape<=20).all())
    assert torch.isfinite(rate).all() and (rate>0).all()
    correction=torch.tensor([[.3,-.2,.4,-.5,-1.,1.]],dtype=torch.float64,requires_grad=True)
    baseline=torch.tensor([3.],dtype=torch.float64)
    area=quad(lambda t:float(head.compute_log_time_pf(correction,[t],baseline).exp().detach()),0,np.inf)[0]
    assert area==pytest.approx(1,abs=1e-9)
    waits,scores=head.sample_and_log_time_pf(correction.expand(200,-1),baseline.expand(200))
    torch.testing.assert_close(scores,head.compute_log_time_pf(correction.expand(200,-1),waits,baseline.expand(200)))
    scores.mean().backward()
    assert torch.isfinite(correction.grad).all() and (correction.grad.abs()>0).all()
    with pytest.raises(ValueError,match='non-finite'):
        head.mixture_parameters(extreme*float('inf'),baseline)


def test_legacy_checkpoint_preserves_policy_semantics(tmp_path):
    g=model(time_parameterization='legacy')
    g.model_kwargs['state_feature_transform']='identity'
    worker=RolloutWorker(g.env)
    with torch.no_grad():
        before,paths=worker.rollout(g,4)
    path=tmp_path/'old.pt';g.save(path)
    data=load_checkpoint(path)
    for key in ('time_parameterization','state_feature_transform'):
        data['metadata']['model'].pop(key)
    for key in ('encoder_lr','flow_encoder_grad_scale','tb_loss_weight','flow_scale_mode'):
        data['metadata']['generator_config'].pop(key)
    restored=generator_from_checkpoint(data)
    with torch.no_grad():after,_=RolloutWorker(restored.env).replay(restored,paths)
    torch.testing.assert_close(before['log_paths_pf'],after['log_paths_pf'],rtol=0,atol=0)
    assert restored.model_kwargs['time_parameterization']=='legacy'


def test_new_optimizer_and_policy_roundtrip(tmp_path):
    g=model(encoder_lr=3e-5,flow_encoder_grad_scale=.1,tb_loss_weight=.1)
    out,paths=RolloutWorker(g.env).rollout(g,4,collect_flows=True)
    g.get_loss_from_rollout_outputs(out).backward();g.opt.step()
    g.save(tmp_path/'new.pt')
    restored=generator_from_checkpoint(load_checkpoint(tmp_path/'new.pt'),optimizer=True)
    assert len(restored.opt.param_groups)==3 and restored.flow_encoder_grad_scale==.1
    assert restored.tb_loss_weight==.1
    with torch.no_grad():
        first,_=RolloutWorker(g.env).replay(g,paths)
        second,_=RolloutWorker(restored.env).replay(restored,paths)
    torch.testing.assert_close(first['log_paths_pf'],second['log_paths_pf'],rtol=0,atol=0)


def test_evaluation_crash_preserves_checkpoint_and_failure_status(tmp_path):
    path,data=fixture_dataset(tmp_path)
    exit_codes=[]
    run=SimpleNamespace(id='failure-test',finish=lambda exit_code:exit_codes.append(exit_code))
    options=dict(dataset_path=str(path),output_path=str(tmp_path/'run'),epochs=2,batch_size=2,
        init_z_sample_count=2,eval_episodes=2,eval_every=1,checkpoint_every=1,
        wandb=True,verbose=False,model_kwargs=dict(embedding_size=16,hidden_size=32,
        transformer_depth=1,transformer_heads=2))
    with patch('train.load_snp_dataset',return_value=data),patch.dict('sys.modules',wandb=SimpleNamespace(init=lambda **kw:run)),\
         patch('train.evaluate_generator',side_effect=AssertionError('injected evaluation failure')):
        with pytest.raises(AssertionError,match='injected'):
            train(**options)
    assert exit_codes==[1]
    failure=json.loads((tmp_path/'run/failure.json').read_text())
    assert failure['phase']=='evaluation' and failure['completed_updates']==1
    assert failure['evaluation']['repeat']==0 and 'injected' in failure['traceback']
    for name in ('latest.pt','checkpoint_0001.pt'):
        checkpoint=load_checkpoint(tmp_path/'run/checkpoints'/name)
        assert checkpoint['trainer']['completed_updates']==1
        assert checkpoint['metadata']['step']==1


def test_wall_time_stop_resumes_exactly(tmp_path,capsys):
    path,data=fixture_dataset(tmp_path)
    options=dict(dataset_path=str(path),epochs=3,batch_size=2,init_z_sample_count=2,
                 verbose=True,replay_fraction=0.,encoder_lr=3e-5,flow_encoder_grad_scale=.1,
                 tb_loss_weight=.25,model_kwargs=dict(embedding_size=16,hidden_size=32,
                 transformer_depth=1,transformer_heads=2))
    with patch('train.load_snp_dataset',return_value=data):
        full,_=train(output_path=str(tmp_path/'full'),**options)
        _,stopped=train(output_path=str(tmp_path/'part'),max_wall_seconds=1e-12,**options)
    assert stopped.completed_updates==1
    record=json.loads((tmp_path/'part/run_status.json').read_text())
    assert record['status']=='interrupted' and record['reason']=='wall_time_budget'
    console=capsys.readouterr().out
    assert 'Stopped after update 1: wall_time_budget.' in console
    assert str(tmp_path/'part/checkpoints/latest.pt') in console
    training_rows=[json.loads(line) for line in (tmp_path/'part/training.jsonl').read_text().splitlines()]
    assert training_rows[-1]['step']==stopped.completed_updates
    with patch('train.load_snp_dataset',return_value=data) as loader:
        resumed,trainer=train(output_path=str(tmp_path/'resume'),epochs=3,verbose=False,
            resume_checkpoint=str(tmp_path/'part/checkpoints/latest.pt'),max_wall_seconds=0.)
    loader.assert_called_once_with(str(path))
    assert trainer.completed_updates==3
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],rtol=0,atol=0)
    assert json.loads((tmp_path/'resume/run_status.json').read_text())['status']=='completed'


def test_resume_inherits_dataset_output_and_device_for_terminal_evaluation(tmp_path):
    path,data=fixture_dataset(tmp_path)
    output=tmp_path/'run'
    options=dict(dataset_path=str(path),output_path=str(output),device='cpu:0',
        epochs=2,batch_size=2,init_z_sample_count=2,eval_episodes=2,eval_every=1,
        terminal_eval=True,terminal_eval_repeats=1,verbose=False,
        model_kwargs=dict(embedding_size=16,hidden_size=32,
                          transformer_depth=1,transformer_heads=2))
    metrics=dict(eval_subtb_loss=1.,eval_log_weight_std=1.,eval_ess=1.)
    with patch('train.load_snp_dataset',return_value=data), \
         patch('train.evaluate_generator',return_value=(metrics,{})), \
         patch('eval.posterior_summary.TerminalSamplingEvaluator.from_dataset') as factory:
        train(max_wall_seconds=1e-12,**options)
        factory.reset_mock()
        _,trainer=train(resume_checkpoint=str(output/'checkpoints/latest.pt'),
                        max_wall_seconds=0.)
    assert trainer.completed_updates==2
    factory.assert_called_once()
    assert factory.call_args.args[0]==str(path)
    checkpoint=load_checkpoint(output/'checkpoints/latest.pt')
    saved=checkpoint['metadata']['resolved_config']
    assert saved['dataset_path']==str(path)
    assert saved['output_path']==str(output)
    assert saved['device']=='cpu:0'
    assert json.loads((output/'run_status.json').read_text())['status']=='completed'


def test_best_checkpoint_uses_calibration_and_reports_initial(tmp_path):
    path,data=fixture_dataset(tmp_path)
    values=iter([(100.,10.),(50.,5.),(1.,8.)])
    def evaluation(*args,**kwargs):
        loss,spread=next(values)
        return dict(eval_subtb_loss=loss,eval_log_weight_std=spread,eval_ess=1.),{}
    with patch('train.load_snp_dataset',return_value=data),patch('train.evaluate_generator',side_effect=evaluation):
        train(dataset_path=str(path),output_path=str(tmp_path/'run'),epochs=2,batch_size=2,
              init_z_sample_count=2,eval_initial=True,eval_episodes=2,eval_every=1,
              verbose=False,model_kwargs=dict(embedding_size=16,hidden_size=32,
              transformer_depth=1,transformer_heads=2))
    checkpoint=load_checkpoint(tmp_path/'run/checkpoints/best_eval.pt')
    assert checkpoint['metadata']['step']==1
    assert checkpoint['metadata']['best_eval_score']==5.
    assert checkpoint['metadata']['best_eval_metric']=='eval_log_weight_std'
    records=[json.loads(l) for l in (tmp_path/'run/evaluation.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==[0,1,2]


@pytest.mark.parametrize('terminal_eval', [False, True])
def test_sampling_repeat_schedule_is_independent_of_truth_evaluation(tmp_path, terminal_eval):
    path,data=fixture_dataset(tmp_path)
    output=tmp_path/'run'
    metrics=dict(eval_subtb_loss=1.,eval_log_weight_std=1.,eval_ess=1.)
    with patch('train.load_snp_dataset',return_value=data), \
         patch('train.evaluate_generator',return_value=(metrics,{})) as evaluate, \
         patch('eval.posterior_summary.TerminalSamplingEvaluator.from_dataset') as truth:
        train(dataset_path=str(path),output_path=str(output),epochs=3,batch_size=2,
              init_z_sample_count=2,eval_episodes=2,eval_every=3,
              terminal_eval=terminal_eval,terminal_eval_repeats=3,
              terminal_eval_repeat_every=2,eval_seed=100007,verbose=False,
              model_kwargs=dict(embedding_size=16,hidden_size=32,
                                transformer_depth=1,transformer_heads=2))
    records=[json.loads(line) for line in (output/'evaluation.jsonl').read_text().splitlines()]
    assert [(r['step'],r['repeat']) for r in records]==[(2,0),(2,1),(2,2),(3,0)]
    assert [call.kwargs['seed'] for call in evaluate.call_args_list]==[102007,102008,102009,103007]
    assert truth.call_count==(2 if terminal_eval else 0)
    if not terminal_eval:
        assert all(call.kwargs['terminal_evaluator'] is None for call in evaluate.call_args_list)
