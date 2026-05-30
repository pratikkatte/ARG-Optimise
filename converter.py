import numpy as np
import tskit
from env import ARGState, CoalescenceChoice, RecombinationChoice

def tree_sequence_to_arg_state(ts: tskit.TreeSequence, env) -> ARGState:
    """
    Convert a tskit.TreeSequence into a valid GFlowNet environment ARGState.
    
    This function:
      1. Detects recombination events by identifying child nodes with multiple parents.
      2. Detects coalescence events by grouping edges by parent node.
      3. Orders events chronologically.
      4. Progressively constructs the ARGState using env.apply_action.
    """
    assert ts.num_samples == env.num_sequences, (
        f"Tree sequence has {ts.num_samples} samples, but environment has {env.num_sequences}."
    )
    
    # 1. Extract recombination events
    recomb_events = []
    for c in range(ts.num_nodes):
        edges = sorted([e for e in ts.edges() if e.child == c], key=lambda e: e.left)
        if not edges:
            continue
        
        breakpoints = []
        current_parent = edges[0].parent
        for e in edges[1:]:
            if e.parent != current_parent:
                # Recombination breakpoint at e.left
                bp_block = int(round(e.left * env.num_blocks / ts.sequence_length))
                if 0 < bp_block < env.num_blocks:
                    breakpoints.append(bp_block)
                current_parent = e.parent
                
        if breakpoints:
            breakpoints = sorted(list(set(breakpoints)))
            c_time = ts.node(c).time / (2.0 * env.population_size)
            parent_times = [ts.node(e.parent).time / (2.0 * env.population_size) for e in edges]
            min_parent_time = min(parent_times)
            
            # Space out multiple recombinations on the same lineage
            r = len(breakpoints)
            for i, bp in enumerate(breakpoints):
                t_event = c_time + (i + 1) / (r + 1) * (min_parent_time - c_time)
                recomb_events.append({
                    'event_type': 'recomb',
                    'ts_node_id': c,
                    'breakpoint': bp,
                    'time': t_event
                })

    # 2. Extract coalescence events
    coal_events = []
    for p in range(ts.num_nodes):
        p_edges = [e for e in ts.edges() if e.parent == p]
        if not p_edges:
            continue
        p_time = ts.node(p).time / (2.0 * env.population_size)
        coal_events.append({
            'event_type': 'coal',
            'ts_node_id': p,
            'time': p_time
        })

    # 3. Sort events chronologically
    events = recomb_events + coal_events
    events.sort(key=lambda x: x['time'])

    # 4. Reconstruct ARGState step-by-step
    state = env.get_initial_state()
    
    # Mapping from environment node_id to tskit node_id
    lineage_id_to_ts_node = {i: i for i in range(env.num_sequences)}

    for event in events:
        if event['event_type'] == 'recomb':
            # Find active lineage that represents this ts_node_id and spans the breakpoint
            bp = event['breakpoint']
            candidates = [
                (idx, lineage) for idx, lineage in enumerate(state.active_lineages)
                if lineage_id_to_ts_node.get(lineage.node_id) == event['ts_node_id']
            ]
            
            target_idx = None
            for idx, lineage in candidates:
                if lineage.material_segments.covers_interval(bp - 1, bp + 1):
                    target_idx = idx
                    break
            
            if target_idx is None:
                # No active lineage spans this breakpoint; it might already be split
                continue
                
            lineage = state.active_lineages[target_idx]
            
            # Calculate time bin
            rates = env.compute_event_rates(env.enumerate_actions(state))
            state.rates = rates
            delta_t = max(1e-10, event['time'] - state.current_time)
            time_action = env.time_env.delta_to_time_action(delta_t, env._total_event_rate(rates))
            
            action = RecombinationChoice(
                active_lineage_i=target_idx,
                material_count=lineage.material_count,
                span_start=lineage.material_segments.span_start,
                span_end=lineage.material_segments.span_end,
                time_action=time_action,
                breakpoint=bp,
                exact_delta_t=delta_t
            )
            
            combined_actions = env.enumerate_actions(state)
            log_prior = env.compute_cwr_event_log_prior(state, combined_actions, action)
            state = env.apply_action(state, action, log_prior=log_prior)
            
            # Update mappings
            parent_id_1 = state.max_node_idx - 1
            parent_id_2 = state.max_node_idx
            lineage_id_to_ts_node[parent_id_1] = event['ts_node_id']
            lineage_id_to_ts_node[parent_id_2] = event['ts_node_id']

        elif event['event_type'] == 'coal':
            p = event['ts_node_id']
            p_edges = [e for e in ts.edges() if e.parent == p]
            
            def get_matching_active_indices():
                indices = []
                for idx, lineage in enumerate(state.active_lineages):
                    ts_child = lineage_id_to_ts_node.get(lineage.node_id)
                    if ts_child is None:
                        continue
                    if ts_child == p:
                        indices.append(idx)
                        continue
                    for e in p_edges:
                        if e.child == ts_child:
                            e_left = int(round(e.left * env.num_blocks / ts.sequence_length))
                            e_right = int(round(e.right * env.num_blocks / ts.sequence_length))
                            overlap_len = lineage.material_segments.intersection_count(
                                [(e_left, e_right)]
                            )
                            if overlap_len > 0:
                                indices.append(idx)
                                break
                return indices
            
            active_indices = get_matching_active_indices()
            
            # Binary coalescence loop
            while len(active_indices) > 1:
                found_pair = False
                for idx1 in range(len(active_indices)):
                    for idx2 in range(idx1 + 1, len(active_indices)):
                        i = active_indices[idx1]
                        j = active_indices[idx2]
                        if state.active_lineages[i].material_segments.overlaps(state.active_lineages[j].material_segments):
                            # Coalesce i and j
                            if i > j:
                                i, j = j, i
                            
                            rates = env.compute_event_rates(env.enumerate_actions(state))
                            state.rates = rates
                            delta_t = max(1e-10, event['time'] - state.current_time)
                            time_action = env.time_env.delta_to_time_action(delta_t, env._total_event_rate(rates))
                            
                            action = CoalescenceChoice(
                                active_lineage_i=i,
                                active_lineage_j=j,
                                time_action=time_action,
                                exact_delta_t=delta_t
                            )
                            
                            combined_actions = env.enumerate_actions(state)
                            log_prior = env.compute_cwr_event_log_prior(state, combined_actions, action)
                            state = env.apply_action(state, action, log_prior=log_prior)
                            
                            # Update mapping
                            lineage_id_to_ts_node[state.max_node_idx] = event['ts_node_id']
                            
                            # Recompute active indices
                            active_indices = get_matching_active_indices()
                            found_pair = True
                            break
                    if found_pair:
                        break
                if not found_pair:
                    break

    return state
