"""Neural migration invariants; long statistical gates live in the acceptance CLI."""
import copy
import math
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import torch
from env.env import SimpleARGEnvironment
from env.snp_data import SNPData, load_snp_dataset
from env.actions import CoalescenceChoice, RecombinationChoice
from generator import GFlowNetGenerator
from policy.observations import pack_states
from gfn.rollout import RolloutWorker, RolloutFailure
from gfn.subtb import geometric_subtb_loss
from training.trainer import Trainer, TrajectoryMixConfig, sample_compatible_trajectories
from training.checkpoints import (load_checkpoint, generator_from_checkpoint, restore_rng,
                                  seed_everything, environment_from_metadata)
from breakpoint_model import SparseMixtureBreakpointPolicy
from infer import validate_terminal, run_inference

REP0 = Path(__file__).resolve().parents[1]/'datasets/sim_5k_mr20/rep0'


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    seed_everything(7)


def environment(n=3, snps=True, recombination=0., length=2):
    genotypes = np.zeros((n,int(snps)),dtype=np.uint8)
    if snps:
        genotypes[0,0]=1
    data=SNPData(genotypes,np.array([.5]) if snps else np.array([]),length,
                 (0,) if snps else (),('A',) if snps else (),('C',) if snps else (),
                 tuple(f'h{i}' for i in range(n)))
    return SimpleARGEnvironment(snp_data=data,population_size=10,mutation_rate=.025,
                                 recombination_rate=recombination)


def model(env=None, **kwargs):
    return GFlowNetGenerator(env or environment(),init_z_sample_count=2,
             model_kwargs=dict(embedding_size=16,hidden_size=32,transformer_depth=1,transformer_heads=2),
             initialize_z_from_policy=False, **kwargs)


def test_features_and_invariant_lineage():
    env=environment(recombination=.01,length=4);state=env.get_initial_state()
    obs=pack_states(env,[state]).observations
    assert obs.snps.shape==(3,13)
    assert obs.snps[0,:3].tolist()==[0.,1.,0.]
    assert obs.snps[1,:3].tolist()==[1.,0.,0.]
    choice=env.enumerate_policy_actions(state)[1][0]
    state=env.apply_action(state,replace(choice,breakpoint=1,delta_t=.2))
    obs=pack_states(env,[state]).observations
    assert obs.snp_lengths[-1]==0
    assert obs.lineage_scalars[-1,-1]==0
    assert obs.interval_lengths[-1]==1
    g=model(env); reps,summary=g.state_encoder(obs)
    assert torch.isfinite(reps).all() and torch.isfinite(summary).all()
    empty=environment(snps=False)
    emptyobs=pack_states(empty,[empty.get_initial_state()]).observations
    assert emptyobs.snps.shape==(0,13)
    assert model(empty).state_encoder(emptyobs)[1].shape==(1,16)


def test_shared_encoder_and_lineage_equivariance():
    env=environment();g=model(env);state=env.get_initial_state()
    permuted=state.clone();permuted.active_lineages=[permuted.active_lineages[i] for i in (2,0,1)]
    _,first,summary=g.encode([state]);_,second,summary2=g.encode([permuted])
    torch.testing.assert_close(first[:,[2,0,1]],second,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(summary,summary2,atol=1e-6,rtol=1e-5)
    parameter_ids=[id(p) for group in g.opt.param_groups for p in group['params']]
    assert len(parameter_ids)==len(set(parameter_ids))==len(list(g.parameters()))
    assert not any('flow_encoder' in name for name,_ in g.named_parameters())
    with torch.no_grad():
        g.flow_head[-1].weight.normal_(0,.1)
        g.arg_model.time_head.output_layer.weight.normal_(0,.1)
    batch,_,summary=g.encode([state])
    g.state_flows([state],summary,batch.observations).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in g.state_encoder.parameters() if p.grad is not None)>0
    g.opt.zero_grad(set_to_none=True)
    g([state],forced_actions=[CoalescenceChoice(1,2,delta_t=.2)])['log_pf'].sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in g.state_encoder.parameters() if p.grad is not None)>0
    before=summary.detach().clone();g.opt.step()
    assert not torch.equal(before,g.encode([state])[2])
    assert env.device=='cpu'


def test_mask_prior_and_factor_normalization():
    env=environment();g=model(env);state=env.get_initial_state()
    output=g([state],forced_actions=[CoalescenceChoice(1,2,delta_t=.25)])
    assert output['factors'][0,0]==0 and output['factors'][0,2]==0
    assert float(output['log_pf'].detach())==pytest.approx(-3*.25,abs=1e-10)
    assert env.compute_cwr_event_log_prior(state,output['actions'][0])==pytest.approx(-.75)
    from validation.tests.test_infinite_sites_environment import environment as fixture_env
    env=fixture_env([[1,0],[1,1],[0,1]],[.5,1.5],length=2)
    g=model(env);state=env.get_initial_state()
    coal,recomb=env.enumerate_policy_actions(state)
    assert coal==[] and recomb
    out=g([state]);assert out['actions'][0].event_type=='recomb'
    assert out['factors'][0,0]==0
    with pytest.raises(ValueError,match='support'):
        g([state],forced_actions=[CoalescenceChoice(0,1,delta_t=.2)])


def test_breakpoint_all_links_and_forced_scores():
    env=environment(recombination=.01,length=8);g=model(env);state=env.get_initial_state()
    choice=env.enumerate_policy_actions(state)[1][0]
    context=torch.zeros(64)
    a,z,params=g.arg_model.breakpoint_head.parameters_for(choice,context,8)
    probs=SparseMixtureBreakpointPolicy.log_probabilities(torch.arange(a,z+1),a,z,params).exp()
    assert float(probs.sum().detach())==pytest.approx(1.,abs=1e-12)
    assert bool((probs>0).all())
    action=replace(choice,breakpoint=6,delta_t=.3)
    first=g([state],forced_actions=[action]);second=g([state],forced_actions=first['actions'])
    torch.testing.assert_close(first['factors'],second['factors'],atol=0,rtol=0)
    assert first['log_pf'].dtype==torch.float64
    state=env.apply_action(state,action)
    # Coalesce disjoint parents with another split to retain a trapped gap.
    g.predecessor(state)
    ancestor,forward=g.predecessor(state)
    assert ancestor.actions==() and forward==action
    restored=env.apply_action(ancestor,forward)
    assert restored.partial_log_likelihood==state.partial_log_likelihood


@pytest.mark.parametrize('lam',[0.,.9,1.,2.])
def test_subtb_loss_and_gradients_against_enumeration(lam):
    pf=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    flows=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    lengths=torch.tensor([3,2]);pb=torch.zeros_like(pf);rewards=flows[torch.arange(2),lengths].detach()
    actual=geometric_subtb_loss(pf,pb,flows,lengths,rewards,lam)
    rows=[]
    for row,length in enumerate(lengths.tolist()):
        terms=[];weights=[]
        for start in range(length):
            for end in range(start+1,length+1):
                weight=lam**(end-start-1)
                residual=flows[row,start]-flows[row,end]+pf[row,start:end].sum()
                terms.append(weight*residual.square());weights.append(weight)
        rows.append(sum(terms)/sum(weights))
    expected=torch.stack(rows).mean()
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    for x,y in zip(torch.autograd.grad(actual,(pf,flows),retain_graph=True),torch.autograd.grad(expected,(pf,flows))):
        torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)


def test_chunked_gradient_matches_direct_shared_gradient():
    env=environment();g=model(env);worker=RolloutWorker(env)
    with torch.no_grad():
        g.flow_head[-1].weight.normal_(0,.03)
        g.arg_model.action_head[-1].weight.normal_(0,.03)
        g.arg_model.time_head.output_layer.weight.normal_(0,.03)
    with torch.no_grad():
        _,paths=worker.rollout(g,episodes=3)
    outputs,_=worker.replay(g,paths)
    loss=g.get_loss_from_rollout_outputs(outputs);loss.backward()
    expected={name:p.grad.clone() if p.grad is not None else torch.zeros_like(p) for name,p in g.named_parameters()}
    g.opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        detached,_=worker.replay(g,paths)
    detached['log_paths_pf'].requires_grad_();detached['state_flows'].requires_grad_()
    weights=torch.autograd.grad(g.get_loss_from_rollout_outputs(detached),
                               (detached['log_paths_pf'],detached['state_flows']))
    worker.backward_scores(g,paths,*weights,chunk_steps=1)
    for name,p in g.named_parameters():
        torch.testing.assert_close(p.grad if p.grad is not None else torch.zeros_like(p),expected[name],atol=2e-5,rtol=2e-5)


def test_terminal_and_unique_backward_replay():
    env=environment();g=model(env);worker=RolloutWorker(env)
    out,paths=worker.rollout(g,episodes=3,collect_flows=True,return_states=True)
    replay,_=worker.replay(g,paths,return_states=True)
    torch.testing.assert_close(out['log_factors'],replay['log_factors'])
    assert torch.count_nonzero(out['log_paths_pb'])==0
    for row,state in enumerate(out['states']):
        validate_terminal(env,state)
        batch,_,summary=g.encode([state])
        assert g.state_flows([state],summary,batch.observations)[0].item()==state.log_reward
        assert out['state_flows'][row,out['lengths'][row]].item()==state.log_reward
        previous,action=g.predecessor(state)
        restored=env.apply_action(previous,action)
        assert restored.partial_log_likelihood==pytest.approx(state.partial_log_likelihood,abs=1e-12)


def test_checkpoint_resume_and_inference_without_dataset(tmp_path):
    env=environment();g=model(env);g.initialize_flow_center();worker=RolloutWorker(env)
    cfg=TrajectoryMixConfig(replay_fraction=.25,replay_min_size=2)
    trainer=Trainer(g,worker,cfg)
    trainer.train_epoch(batch_size=4)
    path=tmp_path/'model.pt';g.save(path,trainer=trainer)
    second=trainer.train_epoch(batch_size=4)
    expected={k:v.clone() for k,v in g.state_dict().items()}
    checkpoint=load_checkpoint(path)
    restored=generator_from_checkpoint(checkpoint,optimizer=True)
    trainer2=Trainer(restored,RolloutWorker(restored.env),cfg)
    trainer2.load_state_dict(checkpoint['trainer']);restore_rng(restored.env,checkpoint['rng'])
    assert trainer2.train_epoch(batch_size=4)==second
    for name,value in restored.state_dict().items():
        torch.testing.assert_close(value,expected[name],atol=0,rtol=0)
    manifest=run_inference(path,tmp_path/'inference',num_args=4)
    assert manifest['summary']['num_completed']==4
    assert manifest['summary']['max_likelihood_error']<1e-9
    assert 'sequences' not in checkpoint['metadata']
    corrupt=copy.deepcopy(checkpoint['metadata']);corrupt['observations']['positions'][0]=.6
    with pytest.raises(ValueError,match='fingerprint'):
        environment_from_metadata(corrupt)
    with pytest.raises(ValueError,match='checkpoint'):
        environment_from_metadata({'sequences':['A','C']})


def test_failure_does_not_discard_histories():
    env=environment();g=model(env)
    with pytest.raises(RolloutFailure,match='limit') as exc:
        RolloutWorker(env,max_events=1).rollout(g)
    assert len(exc.value.histories)==1 and len(exc.value.histories[0])==1


def test_compatible_exploration_density_and_replay():
    env=environment(recombination=.01);g=model(env)
    paths=sample_compatible_trajectories(env,2,max_events=10000)
    assert all(all(math.isfinite(p) for p in path.log_proposals) for path in paths)
    outputs,_=RolloutWorker(env).replay(g,paths)
    assert torch.isfinite(outputs['log_paths_pf']).all()


@pytest.mark.skipif(not REP0.exists(),reason='rep0 unavailable')
def test_rep0_truth_mapping_and_reference_replay():
    from validation.tests.test_infinite_sites_environment import truth_replay
    import json,tskit
    from eval.posterior_summary import TerminalSamplingEvaluator
    env=SimpleARGEnvironment(snp_data=load_snp_dataset(REP0),population_size=100000,
                             mutation_rate=2.5e-8,recombination_rate=1.25e-9)
    meta=json.loads((REP0/'metadata.json').read_text())
    truth=tskit.load(REP0/'sim_5k_mr20.full.trees')
    state,_=truth_replay(env,truth)
    g=model(env)
    outputs,_=RolloutWorker(env).replay(g,[list(state.actions)],return_states=True)
    assert float(outputs['log_rewards'][0].detach())==pytest.approx(state.log_reward,abs=1e-9)
    evaluator=TerminalSamplingEvaluator.from_dataset(REP0,env,grid_size=4)
    assert evaluator.protocol['verified_exported_sites']==95


def test_allowed_coal_normalization_retains_physical_wait_rate():
    from validation.tests.test_infinite_sites_environment import environment as fixture_env
    env=fixture_env(recombination_rate=0)
    state=env.get_initial_state();g=model(env)
    action=CoalescenceChoice(0,1,delta_t=.3)
    output=g([state],forced_actions=[action])
    assert output['factors'][0,:3].tolist()==[0.,0.,0.]
    assert float(output['log_pf'].detach())==pytest.approx(math.log(3)-.9,abs=1e-12)
    assert env.compute_cwr_event_log_prior(state,action)==pytest.approx(-.9,abs=1e-12)
    step=env.sample_compatible_step(state)
    assert step.log_proposal-step.log_prior==pytest.approx(math.log(3),abs=1e-12)


def test_breakpoints_in_trapped_gap_and_snp_free_parent():
    from validation.tests.test_infinite_sites_environment import split,coal
    env=environment(n=2,recombination=.01,length=4)
    state=split(env,env.get_initial_state(),0,1,.1)
    state=split(env,state,2,3,.1)
    state=coal(env,state,1,3,.1)
    assert state.active_lineages[-1].material_segments.segments==((0,1),(3,4))
    choice=next(a for a in env.enumerate_policy_actions(state)[1] if a.active_lineage_i==2)
    assert choice.breakpoint_count==3
    g=model(env)
    action=replace(choice,breakpoint=2,delta_t=.2)
    output=g([state],forced_actions=[action])
    assert torch.isfinite(output['log_pf']).all()
    child=env.apply_action(state,action)
    assert child.active_lineages[-1].snp_indices.size==0
    assert child.active_lineages[-1].material_segments.segments==((3,4),)


def test_shared_encoder_disallows_fixed_feature_warmup():
    from gfn.flow_training import warmup_flow
    with pytest.raises(ValueError,match='shared'):
        warmup_flow(model(),steps=1)


def test_raw_cache_is_exact_and_does_not_cache_age_or_embeddings():
    from policy.observations import RawObservationCache
    env=environment(recombination=.01,length=4);state=env.get_initial_state()
    cache=RawObservationCache()
    first=pack_states(env,[state],cache=cache).observations
    uncached=pack_states(env,[state]).observations
    for field in ('snps','intervals','lineage_scalars','state_scalars'):
        torch.testing.assert_close(getattr(first,field),getattr(uncached,field),atol=0,rtol=0)
    second=env.apply_action(state,CoalescenceChoice(1,2,delta_t=.3))
    cached=pack_states(env,[second],cache=cache).observations
    uncached=pack_states(env,[second]).observations
    assert cache.hits>0
    for field in ('snps','intervals','lineage_scalars','state_scalars'):
        torch.testing.assert_close(getattr(cached,field),getattr(uncached,field),atol=0,rtol=0)
    assert cached.lineage_scalars[0,1]>first.lineage_scalars[0,1]
    assert cache.bytes<=cache.max_bytes


def test_features_decode_large_bitsets_without_float_conversion():
    env=environment(n=70)
    obs=pack_states(env,[env.get_initial_state()]).observations
    assert obs.intervals[69,4+69]==1
    assert obs.intervals[69,4:].sum()==1
    assert obs.snps[69,7+70+69]==1


def test_checkpoint_scheduler_and_compatible_proposal_provenance(tmp_path):
    env=environment(recombination=.001);g=model(env)
    g.scheduler=torch.optim.lr_scheduler.ExponentialLR(g.opt,gamma=.99)
    cfg=TrajectoryMixConfig(exploration_fraction=.25,replay_fraction=.25,replay_min_size=2)
    trainer=Trainer(g,RolloutWorker(env),cfg)
    info=trainer.train_epoch(batch_size=4)
    assert info['compatible_proposal']==1
    entries=list({**trainer.buffer.reservoir,**trainer.buffer.elite}.values())
    proposal=[e for e in entries if e.source=='compatible_proposal']
    assert len(proposal)==1 and math.isfinite(proposal[0].log_proposal)
    checkpoint=g.save(tmp_path/'scheduler.pt',trainer=trainer)
    restored=generator_from_checkpoint(checkpoint,optimizer=True)
    assert restored.scheduler.state_dict()==g.scheduler.state_dict()
    assert [x['lr'] for x in restored.opt.param_groups]==[x['lr'] for x in g.opt.param_groups]
