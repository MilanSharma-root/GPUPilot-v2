from simulator import GPUSimulator
from engine.optimizer import GPUPilotOptimizer
from engine.predictor import GPUPilotPredictor
from engine.sustainability import estimate_impact


def test_simulator_has_data():
    sim = GPUSimulator()
    assert sim.sample_counter == 100
    assert sim.latest.request_rate > 0


def test_simulated_p95_latency_is_present_and_used_for_sla():
    sim = GPUSimulator(seed=123)
    sample = sim.latest
    assert sample.latency_p95_ms > 0
    assert sample.latency_p95_ms >= sample.latency
    assert sample.latency_ok == (sample.latency_p95_ms <= sim.sla.max_latency_ms)


def test_p95_latency_is_used_for_steady_state_sla():
    sim = GPUSimulator(seed=123)
    estimate = sim.estimate_steady_state(sim.configuration)
    assert estimate.latency_p95_ms >= estimate.latency
    assert estimate.sla.p95_latency_ms == estimate.latency_p95_ms
    assert estimate.sla.latency_ok == (estimate.latency_p95_ms <= sim.sla.max_latency_ms)


def test_optimizer_respects_sla():
    sim = GPUSimulator()
    result = GPUPilotOptimizer(sim).optimize()
    assert result.feasible_candidates >= 1
    assert result.recommended_estimate.feasible


def test_prediction_has_output():
    sim = GPUSimulator()
    result = GPUPilotPredictor().predict(sim.get_history(include_live=True, limit=100))
    assert result.predicted_power >= 0
    assert result.predicted_temperature >= 0


def test_environmental_impact_is_estimated():
    sim = GPUSimulator()
    estimate = sim.estimate_steady_state(sim.configuration)
    impact = estimate_impact(estimate)
    assert impact["estimated_water_l_per_hour"] >= 0


def test_cluster_allocation_matches_configuration():
    sim = GPUSimulator()
    sim.apply_configuration(__import__('models').ConfigurationUpdate(gpu_allocation=2))
    cluster = sim.get_cluster()
    assert cluster[0]['active'] or cluster[1]['active']
    assert sum(1 for g in cluster if g['active']) == 2


def test_spike_recovers_automatically():
    sim = GPUSimulator()
    sim.trigger_spike(peak_rate=1150, spike_duration=3, peak_hold=2)
    sim.step(20)
    assert sim.state.value in {'recovery', 'normal'}


def test_optimizer_uses_total_gpu_footprint():
    sim = GPUSimulator()
    opt = GPUPilotOptimizer(sim)
    candidates = opt.generate_candidates()
    assert all(c.gpu_allocation * c.replicas <= 5 for c in candidates)


def test_simulated_quality_gate_passes_when_threshold_allows_degradation():
    from quality import evaluate_simulated_quality
    result = evaluate_simulated_quality("FP16", "INT8", 2.0)
    assert result["evaluation_mode"] == "simulated"
    assert result["result"] == "PASS"
    assert result["quality_degradation_percentage_points"] == 1.3


def test_simulated_quality_gate_rejects_strict_threshold():
    from quality import evaluate_simulated_quality
    result = evaluate_simulated_quality("FP16", "INT8", 0.5)
    assert result["result"] == "FAIL"
    assert "no model inference" in result["disclaimer"]


def test_simulator_random_by_default_and_seed_reproducible():
    a = GPUSimulator(seed=123)
    b = GPUSimulator(seed=123)
    assert [x.request_rate for x in a.initial_dataset] == [x.request_rate for x in b.initial_dataset]


def test_recovery_does_not_exit_against_stale_fixed_baseline():
    """Recovery must converge to the drifting normal demand, not hard-coded 850."""
    from models import WorkloadState

    sim = GPUSimulator(seed=7)
    sim._normal_demand_level = 1120.0
    sim.latent.req = 850.0
    sim.trigger_recovery()
    sim._advance_state()
    assert sim.state == WorkloadState.RECOVERY


def test_automatic_spike_schedule_has_long_irregular_quiet_periods():
    sim = GPUSimulator(seed=123)
    assert 35 <= sim._next_auto_spike_tick <= 100
    intervals = [sim._schedule_next_auto_spike() for _ in range(12)]
    assert all(35 <= value <= 100 for value in intervals)
    assert len(set(intervals)) > 6


def test_live_automatic_spikes_are_not_packed_into_short_repeating_cycles():
    sim = GPUSimulator(seed=321)
    previous = sim.state
    starts = []
    for _ in range(500):
        sample = sim.tick()
        current = sample.workload_state
        if current.value == "spike" and previous.value != "spike":
            starts.append(sample.sample_id)
        previous = current
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) >= 2
    # A burst includes ramp/hold/recovery plus at least 35 NORMAL ticks.
    assert min(gaps) >= 35


def test_controller_uses_demo_responsive_but_bounded_switch_timing():
    """Configuration control should react promptly without switching every tick."""
    from engine.controller import GPUPilotController
    sim = GPUSimulator(seed=5)
    controller = GPUPilotController(sim, GPUPilotOptimizer(sim), GPUPilotPredictor())
    assert controller.interval_s == 2.0
    assert controller.cooldown_s == 3.0
    assert controller.min_improvement_pct == 1.5


def test_recovery_hold_only_blocks_scale_down_while_demand_is_still_elevated():
    """A recovering state alone must not freeze a feasible configuration change."""
    from engine.controller import GPUPilotController
    import inspect
    source = inspect.getsource(GPUPilotController.run_cycle)
    assert 'self.simulator.latest.request_rate >= 950' in source
    assert 'self.simulator.latest.sla_status == "FAIL"' in source


def test_optimizer_explores_broad_configuration_space():
    """All tunable dimensions should vary; total footprint stays within 5 GPUs."""
    sim = GPUSimulator(seed=13)
    candidates = GPUPilotOptimizer(sim).generate_candidates()
    assert len(candidates) >= 300
    assert {c.precision for c in candidates} == {"FP32", "FP16", "INT8"}
    assert {c.batch_size for c in candidates} >= {8, 16, 32, 64}
    assert {c.gpu_allocation for c in candidates} >= {1, 2, 3, 4, 5}
    assert {c.replicas for c in candidates} >= {1, 2, 3}
    assert {c.concurrency for c in candidates} >= {16, 24, 32, 64}
    assert all(c.gpu_allocation * c.replicas <= 5 for c in candidates)


def test_optimizer_gpu_floor_uses_total_gpu_footprint():
    """Replicas contribute to capacity; e.g. 1 GPU x 3 replicas counts as 3."""
    sim = GPUSimulator(seed=14)
    optimizer = GPUPilotOptimizer(sim)
    assert optimizer.required_gpu_count(1800) == 2
    candidates = optimizer.generate_candidates()
    assert any(c.gpu_allocation == 1 and c.replicas == 2 for c in candidates)
