"""Analytic/reference checks for exact local Wasserstein and clade agreement."""
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from paper.scripts.paper_datasets.test_evaluate import feature
from paper.scripts.paper_datasets.figure_3 import (
    empirical_wasserstein_sorted, local_wasserstein, summarize_features, clade_agreement)


@pytest.mark.parametrize('n,m',[(1,1),(2,3),(3,7),(1000,1800)])
def test_vectorized_empirical_w1_matches_scipy(n,m):
    rng=np.random.default_rng(26)
    a=np.sort(rng.uniform(size=(2,3,n)),axis=-1)
    b=np.sort(rng.uniform(size=(2,3,m)),axis=-1)
    actual=empirical_wasserstein_sorted(a,b)
    expected=np.array([[wasserstein_distance(a[i,j],b[i,j]) for j in range(3)] for i in range(2)])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_pair_swapping_is_detected_despite_identical_pooled_distributions():
    a=feature([0,10],[[1,3,4]],[(3,)])
    b=feature([0,10],[[3,1,4]],[(5,)])
    actual,per_pair,_=local_wasserstein([[a],[b],[a]],.5,10)
    np.testing.assert_allclose(actual,[2,0])
    sa,sb=[summarize_features([d],.5,10) for d in (a,b)]
    assert wasserstein_distance(sa['tmrca'],sb['tmrca'],sa['weight'],sb['weight'])==0


def test_local_distance_exact_spans_and_chunk_independence():
    a=[feature([0,2,10],[[1,3],[5,6]],[(),()]),feature([0,5,10],[[2,3],[4,5]],[(),()])]
    b=[feature([0,7,10],[[i+1,i+2],[i+3,i+4]],[(),()]) for i in range(3)]
    expected=0.
    for left,right in zip([0,2,5,7],[2,5,7,10]):
        va=[d.times[d.at(left),0]/4 for d in a]
        vb=[d.times[d.at(left),0]/4 for d in b]
        expected+=(right-left)/10*wasserstein_distance(va,vb)
    for chunk in (1,3,16):
        actual,_,intervals=local_wasserstein([a,b,a],2,10,chunk)
        assert intervals==4
        np.testing.assert_allclose(actual,[expected,0])


def test_pooled_mass_weights_draws_pairs_and_spans():
    a=feature([0,2,10],[[2,2],[6,6]],[(),()])
    b=feature([0,10],[[4,4]],[()])
    summary=summarize_features([a,b],1,10)
    np.testing.assert_allclose(summary['tmrca'],[1,2,3])
    np.testing.assert_allclose(summary['weight'],[.1,.5,.4])


def test_fixed_clade_universe_includes_all_zero_events():
    def ensemble(events):
        return dict(mask=np.array([e[0] for e in events]),position=np.array([e[1] for e in events]),
                    change=np.array([e[2] for e in events]),draws=2)
    a=ensemble([(3,0,2),(3,10,-2)])
    b=ensemble([(3,0,2),(3,2,-2),(5,2,2),(5,10,-2)])
    c=ensemble([(5,0,2),(5,10,-2)])
    fixed=clade_agreement([a,b,c],10,3,'fixed_universe')
    shared=clade_agreement([a,b,c],10,3,'shared_union')
    assert fixed[0][1]['clade_rmse']==pytest.approx(np.sqrt(1.6/3))
    assert shared[0][1]['clade_rmse']==pytest.approx(np.sqrt(.8))
    for i,(points,stats) in enumerate(fixed):
        assert points['weight'].sum()==pytest.approx(1)
        zero=(points['reference_probability']==0)&(points['argflow_probability']==0)
        # For A/B, clade 5 is also absent from both on the first 20% of the
        # genome, even though it belongs to the union through method C.
        assert points['weight'][zero].sum()==pytest.approx(.4 if i==0 else 1/3)
        assert points['weight'][-1]==pytest.approx(1/3)
        error=points['reference_probability']-points['argflow_probability']
        assert np.sqrt(np.dot(points['weight'],error**2))==pytest.approx(stats['clade_rmse'])
