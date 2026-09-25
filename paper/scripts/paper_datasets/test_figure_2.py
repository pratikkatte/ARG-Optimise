"""Numerical checks for the Figure 2 reconstruction and display weights."""
import numpy as np
import pytest

from paper.scripts.paper_datasets.test_evaluate import feature
from paper.scripts.paper_datasets.evaluate import time_metrics
from paper.scripts.paper_datasets.figure_2 import exact_mean, panel_statistics


def test_change_sweep_matches_direct_posterior_mean_and_table_rmse():
    truth = feature([0,2,10], [[2,6,6],[4,8,8]], [(),()])
    draws = [feature([0,3,10], [[1,5,6],[3,7,8]], [(),()]),
             feature([0,7,10], [[5,9,10],[7,11,12]], [(),()])]
    x,y,w = exact_mean(truth,draws,2)
    left = np.array([0,2,3,7])
    np.testing.assert_allclose(x.reshape(-1,2),truth.times[truth.at(left),:-1]/4)
    expected = np.mean([d.times[d.at(left),:-1] for d in draws],axis=0)/4
    np.testing.assert_allclose(y.reshape(-1,2),expected)
    np.testing.assert_allclose(w.reshape(-1,2),np.broadcast_to(np.array([2,1,4,3])[:,None]/20,(4,2)))
    stats,_,_ = panel_statistics(x,y,w,np.geomspace(.01,10,81))
    reference,_,_ = time_metrics(truth,draws,2,['pair_tmrca_rmse'],chunk_size=1)
    assert stats['rmse_2Ne']==pytest.approx(reference['pair_tmrca_rmse'])


def test_underflow_is_accounted_for_and_included_in_statistics():
    x=np.array([.1,1,2.]);y=np.array([.001,1,2.]);w=np.array([.2,.3,.5])
    stats,h,below=panel_statistics(x,y,w,np.geomspace(.01,10,81))
    assert stats['below_range_percent']==pytest.approx(20)
    assert h.sum()==pytest.approx(80)
    assert h.sum()+below.sum()==pytest.approx(100)
    assert stats['rmse_2Ne']==pytest.approx(np.sqrt(.2*.099**2))
    cov=np.cov(np.stack([x,y]),aweights=w,ddof=0)
    assert stats['pearson_r']==pytest.approx(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))


def test_upper_tail_is_never_silently_dropped():
    with pytest.raises(ValueError,match='upper tail'):
        panel_statistics(np.array([1.]),np.array([11.]),np.array([1.]),np.geomspace(.01,10,81))


def test_constant_truth_has_undefined_correlation():
    stats,_,_=panel_statistics(np.array([1.,1.]),np.array([1.,2.]),np.array([.5,.5]),np.geomspace(.01,10,81))
    assert stats['pearson_r'] is None
