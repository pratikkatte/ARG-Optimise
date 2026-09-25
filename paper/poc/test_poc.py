"""Independent distributional and exact-reference checks for the posterior POC."""
import math
import numpy as np
import pytest
import torch
from scipy.integrate import quad

from policy.time_model import CwrGammaMixtureTimeModel, CwrGammaTimeModel
from paper.poc.poc_reference import exact_evidence, sample_prior, trajectory
from paper.poc.poc_metrics import distribution_metrics
from env.env import SimpleARGEnvironment
from env.snp_data import SNPData
from generator import GFlowNetGenerator
from gfn.rollout import RolloutWorker
from training.checkpoints import generator_from_checkpoint, load_checkpoint


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    torch.manual_seed(7)


def test_mixture_integrates_and_scores_marginal_not_selected_component():
    head=CwrGammaMixtureTimeModel(2, 8, 0., components=2)
    corrections=torch.tensor([[math.log(.3),math.log(.7),0.,0.,0.,math.log(2.)]],dtype=torch.float64,requires_grad=True)
    baseline=torch.tensor([3.],dtype=torch.float64)
    # Shapes 1 and 2, both mean 1/3: rates 3 and 6.
    for t in (.01,.2,2.):
        score=head.compute_log_time_pf(corrections,torch.tensor([t],dtype=torch.float64),baseline)
        expected=.3*3*math.exp(-3*t)+.7*36*t*math.exp(-6*t)
        assert float(score.exp().detach()) == pytest.approx(expected, rel=1e-12)
    area=quad(lambda t: float(head.compute_log_time_pf(corrections,torch.tensor([t],dtype=torch.float64),baseline).exp().detach()),0,np.inf)[0]
    assert area == pytest.approx(1.,abs=1e-9)
    score.backward()
    assert bool(torch.isfinite(corrections.grad).all())
    assert bool((corrections.grad.abs()>0).all())


def test_mixture_sampling_moments_and_single_component():
    head=CwrGammaMixtureTimeModel(2,8,0.,components=2)
    corrections=torch.tensor([[math.log(.3),math.log(.7),0.,0.,0.,math.log(2.)]],dtype=torch.float64).expand(30000,-1)
    samples,scores=head.sample_and_log_time_pf(corrections,torch.full((len(corrections),),3.))
    assert float(samples.mean()) == pytest.approx(1/3,abs=.006)
    assert float(samples.var()) == pytest.approx((.3+.7/2)/9,abs=.004)
    torch.testing.assert_close(scores,head.compute_log_time_pf(corrections,samples,torch.full((len(corrections),),3.)))
    one=CwrGammaMixtureTimeModel(2,8,0.,components=1)
    gamma=CwrGammaTimeModel(2,8,0.)
    c=torch.tensor([[0.,.2,.4]],dtype=torch.float64)
    torch.testing.assert_close(one.compute_log_time_pf(c,torch.tensor([.3]),torch.tensor([2.])),
                               gamma.compute_log_time_pf(c[:,1:],torch.tensor([.3]),torch.tensor([2.])))


def test_ctmc_evidence_against_independent_two_locus_laplace_derivative():
    # Differentiating the classical three-state two-locus Laplace transform at
    # a=b=1 and multiplying by kappa^2=1/4 gives this rational value.
    value=exact_evidence(2,.5,.5)
    assert value['evidence'] == pytest.approx(11637/655360,rel=2e-13)
    # With no recombination T1=T2~Exp(1): kappa^2 E[T^2 exp(-4*kappa*T)].
    assert exact_evidence(2,.5,0.)['evidence'] == pytest.approx(.5/27,rel=2e-13)


def environment():
    data=SNPData(np.array([[1,0],[0,1]],dtype=np.uint8),np.array([.5,1.5]),2.,(0,1),('A','A'),('C','C'),('h0','h1'))
    return SimpleARGEnvironment(snp_data=data,population_size=10,mutation_rate=.025,recombination_rate=.025)


def test_independent_prior_histories_match_environment_and_likelihood():
    env=environment(); rng=np.random.default_rng(29)
    for _ in range(100):
        record=sample_prior(rng,2,.5,.5).as_dict()
        state=env.replay(trajectory(record).actions)
        assert state.is_done
        assert state.accumulated_log_prior == pytest.approx(record['log_prior'],abs=1e-10)
        assert env.evaluate_terminal(state).log_likelihood == pytest.approx(record['log_likelihood'],abs=1e-10)
        ts=env.save_to_tree_sequence(state)
        np.testing.assert_allclose([ts.at(x).tmrca(0,1)/20 for x in (.5,1.5)],record['tmrca'],rtol=1e-12)


def test_mixture_replay_gradient_and_checkpoint(tmp_path):
    env=environment()
    model=GFlowNetGenerator(env,init_z_sample_count=2,model_kwargs=dict(embedding_size=16,hidden_size=32,
        transformer_depth=1,transformer_heads=2,continuous_time_head='gamma_mixture',time_mixture_components=3))
    worker=RolloutWorker(env)
    outputs,paths=worker.rollout(model,4,collect_flows=True)
    replay,_=worker.replay(model,paths)
    torch.testing.assert_close(outputs['log_paths_pf'],replay['log_paths_pf'])
    model.get_loss_from_rollout_outputs(outputs).backward()
    assert torch.isfinite(model.arg_model.time_head.output_layer.weight.grad).all()
    path=tmp_path/'model.pt';model.save(path)
    restored=generator_from_checkpoint(load_checkpoint(path))
    rescored,_=worker.replay(restored,paths)
    torch.testing.assert_close(replay['log_paths_pf'],rescored['log_paths_pf'])


def test_tv_identity_on_known_discrete_distributions():
    # Reference p=(.75,.25), policy q=(.25,.75); exact TV=.5.
    rows=[dict(log_prior=0.,log_likelihood=math.log(p),tmrca=[i+1,i+1],recombinations=i)
          for i,p in enumerate((.75,.25))]
    reference=[rows[0]]*3+[rows[1]]
    policy=[rows[0]]+[rows[1]]*3
    q_ref=np.log([.25,.25,.25,.75]);q_policy=np.log([.25,.75,.75,.75])
    metrics=distribution_metrics(reference,policy,q_ref,q_policy,0.)
    assert metrics['full_history_tv']['estimate'] == pytest.approx(.5)
    assert metrics['squared_hellinger']['estimate'] == pytest.approx(1-math.sqrt(.75))


def test_resume_preserves_interrupted_log_and_removes_unsaved_updates(tmp_path):
    from paper.poc.run_poc import reconcile_training_log
    original='{"step": 1}\n{"step": 2}\n{"step": 3}\n'
    log=tmp_path/'training.jsonl';log.write_text(original)
    reconcile_training_log(tmp_path,2)
    assert log.read_text()=='{"step": 1}\n{"step": 2}\n'
    assert next((tmp_path/'interrupted_logs').iterdir()).read_text()==original
    reconcile_training_log(tmp_path,2)
    assert len(list((tmp_path/'interrupted_logs').iterdir()))==1
    with pytest.raises(ValueError,match='does not cover'):
        reconcile_training_log(tmp_path,3)
