"""Read-only sufficient termination/normalization audit for the saved policies.

This bounds the ideal real-arithmetic policy. Numerical overflow, probability
underflow, and likelihood floors remain separate implementation limitations.
"""
import math
import numpy as np
from evaluate_subtb_density_fit import load_checkpoint, read_json, write_json, digest


def check_material_history(events, samples, blocks):
    """Independent integer-bitset reconstruction of the absorption invariant."""
    full=(1<<blocks)-1
    active={i:full for i in range(samples)}
    material=samples*blocks
    nonoverlap=0
    for event in events:
        children=[active.pop(i) for i in event['child_ids']]
        if event['event_type']=='coal':
            overlap=(children[0]&children[1]).bit_count()
            nonoverlap += overlap==0
            active[event['parent_ids'][0]]=children[0]|children[1]
            material -= overlap
        else:
            mask=(1<<event['breakpoint'])-1
            left,right=children[0]&mask,children[0]&~mask
            assert left and right
            active[event['parent_ids'][0]]=left
            active[event['parent_ids'][1]]=right
        assert blocks <= material <= samples*blocks
        assert len(active)<=material and material==sum(v.bit_count() for v in active.values())
        coverage=0
        for value in active.values(): coverage|=value
        assert coverage==full
    assert material==blocks
    return nonoverlap


def audit(output):
    p=read_json(output/'protocol.json')
    rows=[]
    for job in p['jobs']:
        assert digest(job['checkpoint']) == job['sha256']
        checkpoint=load_checkpoint(job['checkpoint'],map_location='cpu')
        meta=checkpoint['metadata'];state=checkpoint['generator_state_dict']
        def array(name):
            value=state['arg_model.'+name].double().cpu().numpy()
            assert np.isfinite(value).all()
            return value
        gamma,beta=array('encoder.norm.weight'),array('encoder.norm.bias')
        dimension=len(gamma)
        rep=np.abs(gamma)*math.sqrt(dimension)+np.abs(beta)
        features=np.concatenate([2*rep,2*rep,rep**2,rep])
        hidden=np.abs(array('action_scorer.0.weight'))@features+np.abs(array('action_scorer.0.bias'))
        bound=float((np.abs(array('action_scorer.3.weight'))@hidden+np.abs(array('action_scorer.3.bias'))).item())
        samples=len(meta['sequences']) if 'sequences' in meta else meta['num_sequences']
        blocks=meta['num_blocks'];maximum_material=samples*blocks
        pairs=maximum_material*(maximum_material-1)//2
        log_lower=-2*bound-math.log(pairs)
        assert math.isfinite(log_lower)
        rows.append(dict(label=job['label'],step=job['step'],checkpoint_sha256=job['sha256'],
            samples=samples,blocks=blocks,
            maximum_active_lineages=maximum_material,maximum_progress_events=maximum_material-blocks,
            pair_logit_absolute_bound=bound,overlap_choice_log_probability_lower_bound=log_lower))
    result=dict(ideal_real_arithmetic_terminal_policy_normalized=True,records=rows,
        assumptions=['Finite checkpoint parameters; finite valid arithmetic on each finite state.',
            'Every active lineage carries at least one of the finite integer genomic blocks.',
            'Recombination partitions nonempty material, preserving total active material; coalescence takes its union.',
            'All Hudson lineage pairs remain legal. Pair logits use bounded final-LayerNorm representations only.',
            'Categorical choices and positive-rate/shape Gamma waits are interpreted as their intended normalized distributions.'],
        argument=[
            'Total active material M starts at n*L, never increases, and is bounded below by L. Thus the active lineage count is at most n*L.',
            'An infinite event sequence must contain infinitely many coalescences: recombination increases lineage count by one and coalescence decreases it by one.',
            'Whenever M>L, at least one pair overlaps. Every such pair is a legal Hudson coalescence.',
            'Final LayerNorm bounds each representation coordinate by abs(gamma)*sqrt(d)+abs(beta). Sum/difference/product action features and the finite ReLU MLP therefore give a uniform finite pair-logit bound B.',
            'Conditional on a coalescence, an overlapping pair is selected with probability at least exp(-2B)/choose(n*L,2), uniformly over event times and histories.',
            'Each overlap reduces integer M by at least one. At most n*L-L progress events are needed. Infinitely many coalescences without absorption consequently have probability zero.',
            'Each Gamma wait is finite almost surely on a finite state; hence the finite absorbing trajectory has finite event times. The prior satisfies the same progress argument with uniform pair choice.'],
        limitations=['This is not finite enumeration of full histories: event counts remain unbounded and event times continuous.',
            'The conservative lower bound is not a useful runtime or convergence bound.',
            'This ideal-distribution proof does not remove finite-precision underflow/overflow, padded-logit approximations, or likelihood-floor effects; independent evaluated-path checks are still required.'])
    write_json(output/'normalization_audit.json',result)
    return result
