import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from arg.env import CoalescenceChoice, SimpleARGEnvironment
from arg.time_env import ContinuousCoalescentTime
from arg.infer import validate_metadata
from arg.rollout_worker_arg import RolloutWorker
from arg.tb_gfn import TBGFlowNetGenerator
from arg.train import MODEL_VERSION
from arg.time_model import BernsteinBetaTimeModel


@pytest.mark.parametrize(
    "rate,max_delta,quantile",
    [
        (1e-4, None, 1e-10),
        (1.0, None, 0.5),
        (1e4, None, 1.0 - 1e-10),
        (1e-4, 1e-8, 0.25),
        (3.0, 0.01, 0.75),
        (1e4, 10.0, 1.0 - 1e-10),
    ],
)
def test_continuous_cwr_quantile_round_trip(rate, max_delta, quantile):
    clock = ContinuousCoalescentTime()
    delta = clock.quantile_to_delta(
        quantile,
        rate,
        max_delta=max_delta,
    )

    assert delta > 0.0
    if max_delta is not None:
        assert delta < max_delta
    assert clock.delta_to_quantile(
        delta,
        rate,
        max_delta=max_delta,
    ) == pytest.approx(quantile, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    "rate,delta",
    [(1e-8, 1e-8), (0.25, 2.0), (1e8, 1e-8)],
)
def test_continuous_cwr_cdf_inverse_cdf_round_trip(rate, delta):
    clock = ContinuousCoalescentTime()
    probability = clock.cdf(delta, rate)

    assert clock.inverse_cdf(probability, rate) == pytest.approx(
        delta,
        rel=1e-12,
        abs=1e-18,
    )


@pytest.mark.parametrize("rate,max_delta", [(0.01, 1e-8), (3.0, 0.2), (100.0, 10.0)])
def test_bounded_generated_and_survival_masses_are_exact(rate, max_delta):
    clock = ContinuousCoalescentTime()
    generated, survival = clock.bounded_waiting_distribution(
        rate,
        max_delta,
    )

    assert generated + survival == pytest.approx(1.0, abs=1e-14)
    assert generated == pytest.approx(
        1.0 - math.exp(-rate * max_delta)
    )
    assert clock.survival_log_probability(
        rate,
        max_delta,
    ) == pytest.approx(-rate * max_delta)


def test_equal_bernstein_weights_are_the_uniform_cwr_quantile_density():
    model = BernsteinBetaTimeModel(
        input_dim=4,
        hidden_dim=8,
        dropout=0.0,
        basis_components=16,
        layers=1,
    )
    logits = model(torch.zeros(101, 4))
    quantiles = torch.linspace(
        1e-8,
        1.0 - 1e-8,
        101,
        dtype=torch.float64,
    )
    log_density = model.log_quantile_density(logits, quantiles)

    assert torch.allclose(
        log_density,
        torch.zeros_like(log_density),
        atol=1e-10,
        rtol=1e-10,
    )


def test_arbitrary_bernstein_density_normalizes_and_has_finite_gradients():
    model = BernsteinBetaTimeModel(
        input_dim=4,
        hidden_dim=8,
        dropout=0.0,
        basis_components=16,
        layers=1,
    )
    quantiles = torch.linspace(
        1e-7,
        1.0 - 1e-7,
        20_001,
        dtype=torch.float64,
    )
    logits = torch.linspace(
        -2.0,
        2.0,
        16,
        dtype=torch.float32,
    )[None, :].expand(quantiles.numel(), -1).clone()
    logits.requires_grad_(True)
    density = torch.exp(model.log_quantile_density(logits, quantiles))
    integral = torch.trapezoid(density, quantiles)
    integral.backward()

    assert integral.item() == pytest.approx(1.0, abs=2e-5)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_time_density_includes_exact_conditional_cdf_jacobian():
    clock = ContinuousCoalescentTime()
    model = BernsteinBetaTimeModel(
        input_dim=4,
        hidden_dim=8,
        dropout=0.0,
        basis_components=16,
        layers=0,
    )
    rate = 3.0
    horizon = 0.4
    generated = clock.generated_probability(rate, horizon)
    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    deltas = [
        clock.quantile_to_delta(value, rate, horizon)
        for value in quantiles.tolist()
    ]
    logits = model(torch.zeros(3, 4))

    observed = model.log_time_density(
        logits,
        quantiles,
        deltas,
        [rate] * 3,
        [generated] * 3,
    )
    expected = torch.tensor(
        [
            clock.waiting_time_log_density(delta, rate, horizon)
            - math.log(generated)
            for delta in deltas
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_jacobian_adjusted_bounded_density_integrates_in_physical_time():
    clock = ContinuousCoalescentTime()
    model = BernsteinBetaTimeModel(
        input_dim=4,
        hidden_dim=8,
        dropout=0.0,
        basis_components=16,
        layers=0,
    )
    rate = 2.7
    horizon = 0.8
    generated = clock.generated_probability(rate, horizon)
    deltas = torch.linspace(
        1e-9,
        horizon - 1e-9,
        30_001,
        dtype=torch.float64,
    )
    quantiles = torch.as_tensor(
        [
            clock.delta_to_quantile(delta, rate, horizon)
            for delta in deltas.tolist()
        ],
        dtype=torch.float64,
    )
    logits = torch.linspace(
        -1.5,
        1.5,
        16,
    )[None, :].expand(deltas.numel(), -1)
    log_density = model.log_time_density(
        logits,
        quantiles,
        deltas,
        [rate] * deltas.numel(),
        [generated] * deltas.numel(),
    )

    assert torch.trapezoid(
        torch.exp(log_density),
        deltas,
    ).item() == pytest.approx(1.0, abs=2e-6)


def test_uniform_quantiles_reproduce_exponential_prior_mean():
    clock = ContinuousCoalescentTime()
    rng = random.Random(7)
    rate = 5.0
    samples = [
        clock.quantile_to_delta(
            clock.sample_prior_quantile(rng),
            rate,
        )
        for _ in range(50_000)
    ]

    assert sum(samples) / len(samples) == pytest.approx(
        1.0 / rate,
        rel=0.02,
    )


def test_global_prior_rollout_step_matches_wait_and_event_type_moments():
    np.random.seed(17)
    env = SimpleARGEnvironment(
        num_sequences=4,
        sequence_length=4,
        num_blocks=4,
        rho=3.0,
        seed=17,
        structural_only=True,
    )
    state = env.get_initial_state()
    probabilities = env.compute_event_probabilities(state)
    total_rate = (
        state.rates["lambda_coal"] + state.rates["lambda_recomb"]
    )
    samples = [env._sample_prior_step(state)[0] for _ in range(20_000)]
    coalescence_rate = sum(
        isinstance(action, CoalescenceChoice)
        for action in samples
    ) / len(samples)
    mean_delta = sum(action.delta_time for action in samples) / len(samples)

    assert coalescence_rate == pytest.approx(
        probabilities["coal"],
        abs=0.015,
    )
    assert mean_delta == pytest.approx(1.0 / total_rate, rel=0.025)


def test_continuous_prior_rollout_exports_strict_parent_child_times():
    env = SimpleARGEnvironment(
        sequences=["AAA", "AAA", "AAA", "AAA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        seed=29,
        device="cpu",
    )
    state = env.get_initial_state()
    event_times = []
    while not state.is_done:
        action, log_prior = env._sample_prior_step(state)
        state = env.apply_action(state, action, log_prior=log_prior)
        event_times.append(float(state.current_time))

    tree_sequence = env.save_to_tree_sequence(state)
    tables = tree_sequence.tables
    assert len(set(event_times)) == len(event_times)
    assert np.all(
        tables.nodes.time[tables.edges.parent]
        > tables.nodes.time[tables.edges.child]
    )


def test_zero_horizon_and_invalid_times_are_rejected():
    clock = ContinuousCoalescentTime()

    assert clock.generated_probability(2.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="no continuous"):
        clock.quantile_to_delta(0.5, 2.0, max_delta=0.0)
    with pytest.raises(ValueError, match="strictly before"):
        clock.delta_to_quantile(0.1, 2.0, max_delta=0.1)
    assert clock.waiting_time_log_density(
        0.1,
        2.0,
        max_delta=0.1,
    ) == -math.inf
    with pytest.raises(ValueError, match="positive"):
        clock.cdf(1.0, 0.0)


def test_extreme_bounded_quantile_remains_strictly_inside_horizon():
    clock = ContinuousCoalescentTime()
    horizon = 1e-12
    quantile = math.nextafter(1.0, 0.0)

    delta = clock.quantile_to_delta(quantile, 1e-4, horizon)

    assert 0.0 < delta < horizon


def test_time_context_features_avoid_rate_horizon_product_underflow():
    features = BernsteinBetaTimeModel.context_features(
        [1e-300],
        [1e-300],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.isfinite(features).all()
    assert features.tolist()[0] == pytest.approx(
        [-30.0, 1.0, -30.0, -30.0]
    )


def test_v2_checkpoint_metadata_is_accepted_and_v1_bins_are_rejected():
    metadata = {
        "num_sequences": 2,
        "sequence_length": 2,
        "num_blocks": 2,
        "rho": 0.1,
        "time_scheme": "ContinuousCWRConditionalCDF",
        "time_density": "BernsteinBeta",
        "time_basis_components": 16,
        "time_reference_measure": "delta_t_over_2Ne",
        "demography_model": "constant_ne",
        "seed": 7,
        "init_z_sample_count": 0,
        "model_version": MODEL_VERSION,
        "input_mode": "dense",
        "sequences": ["AA", "AA"],
        "model": {"time_basis_components": 16},
    }

    validate_metadata(metadata)
    with pytest.raises(ValueError, match="must be retrained"):
        validate_metadata(
            {
                **metadata,
                "time_bins": 32,
                "time_delta_bin_width": 0.001,
            }
        )


@pytest.mark.parametrize("loss_mode", ["tb", "subtb", "fl_subtb"])
def test_gfn_losses_backpropagate_into_continuous_mixture_logits(loss_mode):
    env = SimpleARGEnvironment(
        sequences=["AA", "AA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=13,
        device="cpu",
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode=loss_mode,
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
            "breakpoint_gap_layers": 1,
        },
    )
    outputs, _ = RolloutWorker(env).rollout(
        generator,
        episodes=4,
        return_states=True,
    )
    loss = generator.get_loss_from_rollout_outputs(outputs)
    loss.backward()
    mixture_gradient = generator.time_model.output_layer.bias.grad

    assert torch.isfinite(loss)
    assert mixture_gradient is not None
    assert torch.isfinite(mixture_gradient).all()
    assert torch.any(mixture_gradient != 0.0)


def test_backward_reconstruction_recovers_continuous_quantile_and_parent_count():
    env = SimpleARGEnvironment(
        sequences=["AA", "AA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        seed=19,
        device="cpu",
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
            "breakpoint_gap_layers": 1,
        },
    )
    initial_state = env.get_initial_state()
    action, log_prior = env._sample_prior_step(initial_state)
    child_state = env.apply_action(
        initial_state,
        action,
        log_prior=log_prior,
    )
    inverse_actions = generator._enumerate_inverse_arg_actions(child_state)

    assert generator.count_backward_parents(child_state) == len(
        inverse_actions
    ) == 1
    parent_state, reconstructed = generator._apply_inverse_arg_action(
        child_state,
        inverse_actions[0],
    )
    assert (
        parent_state.structural_identity()
        == initial_state.structural_identity()
    )
    assert reconstructed["delta_time"] == pytest.approx(action.delta_time)
    assert reconstructed["time_quantile"] == pytest.approx(
        action.time_quantile,
        rel=1e-12,
        abs=1e-12,
    )


def test_recombination_backward_reconstruction_recovers_continuous_quantile():
    env = SimpleARGEnvironment(
        sequences=["AAAA", "AAAA"],
        bp_per_blocks=1,
        rho=4.0,
        seed=23,
        device="cpu",
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
            "breakpoint_gap_layers": 1,
        },
    )
    initial_state = env.get_initial_state()
    options = env.enumerate_prior_options(initial_state)
    total_rate = (
        options.rates["lambda_coal"]
        + options.rates["lambda_recomb"]
    )
    quantile = 0.37
    action = replace(
        options.recomb_choices[0],
        breakpoint=options.recomb_choices[0].span_start + 1,
        time_quantile=quantile,
        delta_time=env.time_env.quantile_to_delta(
            quantile,
            total_rate,
        ),
    )
    log_prior = env.compute_cwr_event_log_prior(
        initial_state,
        (
            list(options.coal_actions),
            list(options.recomb_choices),
        ),
        action,
        rates=options.rates,
    )
    child_state = env.apply_action(
        initial_state,
        action,
        log_prior=log_prior,
    )
    inverse_actions = generator._enumerate_inverse_arg_actions(child_state)

    assert generator.count_backward_parents(child_state) == len(
        inverse_actions
    ) == 1
    parent_state, reconstructed = generator._apply_inverse_arg_action(
        child_state,
        inverse_actions[0],
    )
    assert (
        parent_state.structural_identity()
        == initial_state.structural_identity()
    )
    assert reconstructed["delta_time"] == pytest.approx(action.delta_time)
    assert reconstructed["time_quantile"] == pytest.approx(
        quantile,
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_accelerator_time_density_has_finite_nonzero_gradients(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device_type == "mps" and not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS is unavailable")

    model = BernsteinBetaTimeModel(
        input_dim=4,
        hidden_dim=8,
        dropout=0.0,
        basis_components=16,
        layers=1,
    ).to(device_type)
    features = torch.randn(8, 4, device=device_type)
    logits = model(features)
    quantiles = model.sample(logits)
    clock = ContinuousCoalescentTime()
    deltas = [
        clock.quantile_to_delta(float(quantile), 2.0)
        for quantile in quantiles.tolist()
    ]
    log_density = model.log_time_density(
        logits,
        quantiles,
        deltas,
        [2.0] * len(deltas),
        [1.0] * len(deltas),
    )
    (-log_density.mean()).backward()
    gradient = model.output_layer.weight.grad

    assert torch.isfinite(log_density).all()
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient != 0.0)


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_accelerator_gfn_rollout_and_backward_smoke(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device_type == "mps" and not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS is unavailable")

    env = SimpleARGEnvironment(
        sequences=["AA", "AA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=11,
        device=device_type,
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device=device_type,
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="subtb",
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
            "breakpoint_gap_layers": 1,
        },
    )
    outputs, _ = RolloutWorker(env).rollout(
        generator,
        episodes=2,
        return_states=True,
    )
    loss = generator.get_loss_from_rollout_outputs(outputs)
    loss.backward()
    time_gradients = [
        parameter.grad
        for parameter in generator.time_model.parameters()
        if parameter.grad is not None
    ]

    assert torch.isfinite(loss)
    assert outputs["time_quantiles"].numel() == 2
    assert time_gradients
    assert all(torch.isfinite(gradient).all() for gradient in time_gradients)
