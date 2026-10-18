"""
simulator.py - workload-driven, correlated GPU simulator.

*** SIMULATED GPU TELEMETRY *** - not readings from a physical NVIDIA GPU.

Causal chain (see README for the full formulas):

    workload state -> target request rate -> request_rate (smoothed, noisy)
    request_rate / serving_capacity(config) = rho  (load factor)
    rho -> utilization -> power -> temperature (smoothed, has inertia)
    rho -> latency (queueing curve) ; min-ish(request, capacity) -> throughput
    config + rho -> memory

All "target" functions below are pure and noise-free. The tick adds smoothing
(dependence on the previous sample) and small noise on top.

CLI:  python simulator.py --export-dir data     # writes the 100-sample JSON + CSV
"""
import argparse
import csv
import io
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

import numpy as np

import config as cfg
from models import (Configuration, ConfigurationUpdate, SLAConfig, SLAResult,
                    SteadyStateEstimate, TelemetrySample, WorkloadState)
from cluster import build_cluster


# ====================================================================== helpers
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp_metric(name: str, x: float) -> float:
    lo, hi = cfg.BOUNDS[name]
    return clamp(x, lo, hi)


def soft_cap(x: float, knee: float, limit: float) -> float:
    """Identity below `knee`, then smoothly saturates toward `limit` (no hard flat-top)."""
    if x <= knee:
        return x
    span = limit - knee
    return knee + span * math.tanh((x - knee) / span)


# ================================================ pure, noise-free model pieces
def serving_capacity(c: Configuration, temp_c: float = 0.0) -> float:
    """Requests/min the deployment can serve.

    capacity = BASE * precision_speed * batch_f * concurrency_f * gpu_f * replica_f * thermal_derate
      batch_f       : (batch/16)^0.30 up to 64, weaker (^0.10) beyond - diminishing returns
      concurrency_f : +10 %/doubling up to 64, then contention penalty
      gpu_f         : gpu_allocation^0.85  (tensor-parallel overhead)
      replica_f     : replicas^0.95        (near-linear horizontal scaling)
      thermal_derate: -1 % per degree above 82 C (throttling)
    """
    speed = cfg.PRECISION[c.precision].speed
    batch_f = (min(c.batch_size, 64) / 16) ** 0.30 * (max(c.batch_size, 64) / 64) ** 0.10
    x = math.log2(c.concurrency / 32)
    conc_f = 1 + 0.10 * x - 0.06 * max(0.0, x - 1) ** 2
    gpu_f = c.gpu_allocation ** 0.85
    rep_f = c.replicas ** 0.95
    derate = 1 - 0.01 * max(0.0, temp_c - 82.0)
    return cfg.BASE_CAPACITY * speed * batch_f * max(conc_f, 0.3) * gpu_f * rep_f * derate


def unloaded_latency(c: Configuration) -> float:
    """Service latency (ms) with an empty queue.

    Bigger batch -> slightly slower per request; faster precision / more GPUs per
    replica -> faster; higher concurrency -> a little more per-request delay.
    """
    speed = cfg.PRECISION[c.precision].speed
    batch_l = max(0.8, 1 + 0.008 * (c.batch_size - 16))
    conc_l = max(0.5, 1 + 0.08 * math.log2(c.concurrency / 32))
    return cfg.BASE_LATENCY_MS * batch_l * conc_l / (speed ** 0.7) / (c.gpu_allocation ** 0.30)


def latency_target(c: Configuration, rho: float) -> float:
    """latency = L0 * (1 + A * rho^4), softly saturating above 200 ms toward 300 ms (gentle knee, no flat-top)."""
    raw = unloaded_latency(c) * (1 + cfg.QUEUE_A * min(rho, 1.6) ** 4)
    return raw if raw <= 200 else 200 + 100 * math.tanh((raw - 200) / 200)


def utilization_target(c: Configuration, rho: float) -> float:
    """util = 18 + 75*rho (+4 per batch doubling: fuller batches keep the GPU busier)."""
    raw = 18 + 75 * min(rho, 1.3) + 4 * math.log2(max(c.batch_size, 1) / 16)
    return soft_cap(raw, 90.0, 98.0)


def power_from_util(c: Configuration, util: float) -> float:
    """Per-GPU power = idle + dynamic * (util/82)^1.3 * precision_factor."""
    dyn = cfg.DYNAMIC_POWER_W * (max(util, 0.0) / 82.0) ** cfg.POWER_EXPONENT
    return cfg.IDLE_POWER_W + dyn * cfg.PRECISION[c.precision].power


def temperature_target(power_w: float) -> float:
    """Where temperature would settle if power stayed constant."""
    return cfg.AMBIENT_C + cfg.THERMAL_RESISTANCE * power_w


def memory_target(c: Configuration, rho: float) -> float:
    """Per-GPU memory (GB) = (weights/gpus + batch acts + concurrency KV-cache + load term) * precision."""
    m = cfg.PRECISION[c.precision].memory
    weights = 8.0 / c.gpu_allocation
    acts = 0.12 * c.batch_size
    kv = 0.03 * c.concurrency
    load = 3.7 * min(rho, 1.1)
    return (weights + acts + kv + load) * m


def throughput_target(req: float, capacity: float) -> float:
    """Smooth-min(request, capacity): follows demand, then flattens at capacity."""
    p = cfg.SOFTMIN_P
    return req * capacity / (req ** p + capacity ** p) ** (1.0 / p)


def _latency_spread(rho: float) -> float:
    """Synthetic request-latency spread; tail variance grows as the service saturates."""
    return min(0.45, 0.12 + 0.22 * max(0.0, rho - 0.65))


def estimate_p95_latency(latency_ms: float, rho: float) -> float:
    """Analytic P95 estimate for a log-normal request-latency distribution."""
    sigma = _latency_spread(rho)
    return max(0.0, latency_ms) * math.exp(1.6448536269514722 * sigma - 0.5 * sigma * sigma)


def sample_p95_latency(latency_ms: float, rho: float, rng: np.random.Generator,
                        sample_count: int = 128) -> float:
    """Generate request-level simulated latencies and calculate their empirical P95."""
    sigma = _latency_spread(rho)
    median = max(latency_ms, 0.1)
    requests = rng.lognormal(mean=math.log(median), sigma=sigma, size=sample_count)
    return float(np.percentile(requests, 95))


def check_sla(sla: SLAConfig, latency_p95: float, throughput: float, temp: float) -> SLAResult:
    l_ok = latency_p95 <= sla.max_latency_ms
    t_ok = throughput >= sla.min_throughput_req_min
    c_ok = temp <= sla.max_temperature_c
    return SLAResult(p95_latency_ms=round(latency_p95, 1), latency_ok=l_ok,
                     throughput_ok=t_ok, temperature_ok=c_ok,
                     sla_status="PASS" if (l_ok and t_ok and c_ok) else "FAIL")


# ================================================================== simulator
@dataclass
class _Latent:
    """Hidden 'true' state carried between ticks (observed values add noise)."""
    req: float
    util: float
    mem: float
    temp: float
    lat: float
    thr: float


class GPUSimulator:
    COMPARE_KEYS = ["request_rate", "gpu_utilization", "gpu_power", "total_power",
                    "temperature", "memory", "latency", "latency_p95_ms", "throughput"]

    def __init__(self, seed: Optional[int] = None):
        # Default runs are intentionally varied; pass an integer seed for reproducible tests.
        self.seed = seed
        self.reset()

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.configuration = Configuration(**cfg.DEFAULT_CONFIGURATION)
        self.sla = SLAConfig(**cfg.DEFAULT_SLA)
        b = cfg.BASELINE
        self.latent = _Latent(b["request_rate"], b["gpu_utilization"], b["memory"],
                              b["temperature"], b["latency"], b["throughput"])
        self.history: Deque[TelemetrySample] = deque(maxlen=cfg.HISTORY_MAXLEN)
        self.sample_counter = 0
        self.state = WorkloadState.NORMAL
        self.state_ticks = 0
        self.manual_rate: Optional[float] = None
        self.peak_rate = cfg.DEFAULT_PEAK_RATE
        self.spike_duration = cfg.DEFAULT_SPIKE_TICKS
        self.peak_hold: Optional[int] = cfg.DEFAULT_PEAK_HOLD_TICKS
        self.recovery_from = cfg.NORMAL_RATE
        # Per-event dynamics prevent every burst/recovery from having the same shape.
        self._active_spike_tau = float(cfg.SPIKE_TAU)
        self._active_recovery_tau = float(cfg.RECOVERY_TAU)
        self._active_peak_sag = float(self.rng.uniform(0.015, 0.095))
        self._active_peak_wobble = float(self.rng.uniform(0.008, 0.045))
        self._active_ramp_exponent = float(self.rng.uniform(0.72, 1.35))
        self._before: Optional[Dict[str, float]] = None
        self._before_config: Optional[Configuration] = None
        self._change_id: Optional[int] = None
        self.auto_spike_enabled = cfg.AUTO_SPIKE_ENABLED
        self._normal_ticks_since_event = 0
        self._auto_spike_index = 0
        self._normal_demand_level = float(cfg.NORMAL_RATE)
        self._normal_demand_target = float(cfg.NORMAL_RATE)
        self._next_demand_drift_tick = int(self.rng.integers(5, 14))
        self._next_auto_spike_tick = self._schedule_next_auto_spike()
        self._generate_initial_dataset()
        # Start live simulation from a clean NORMAL window. Automatic live events
        # use a randomized schedule unless the caller supplied an explicit seed.
        self._normal_ticks_since_event = 0
        self._auto_spike_index = 0
        self._next_auto_spike_tick = self._schedule_next_auto_spike()

    def _schedule_next_auto_spike(self) -> int:
        """Choose a genuinely irregular quiet period between autonomous bursts.

        The former 10–40 tick schedule was close to the combined ramp/hold/recovery
        duration, so a new burst often started immediately after the previous one.
        That created a visually periodic spike/recovery waveform even with noisy peaks.
        """
        return int(self.rng.integers(35, 101))

    # ------------------------------------------------- workload state machine
    def _set_state(self, state: WorkloadState) -> None:
        self.state = state
        self.state_ticks = 0
        self.manual_rate = None
        if state != WorkloadState.NORMAL:
            self._normal_ticks_since_event = 0

    def trigger_spike(self, peak_rate: float = cfg.DEFAULT_PEAK_RATE,
                      spike_duration: int = cfg.DEFAULT_SPIKE_TICKS,
                      peak_hold: Optional[int] = None) -> None:
        self._set_state(WorkloadState.SPIKE)
        self.peak_rate = peak_rate
        self.spike_duration = spike_duration
        self.peak_hold = cfg.DEFAULT_PEAK_HOLD_TICKS if peak_hold is None else peak_hold
        self._active_spike_tau = float(cfg.SPIKE_TAU * self.rng.uniform(0.55, 1.65))
        self._active_peak_sag = float(self.rng.uniform(0.015, 0.095))
        self._active_peak_wobble = float(self.rng.uniform(0.008, 0.045))
        self._active_ramp_exponent = float(self.rng.uniform(0.72, 1.35))

    def trigger_recovery(self) -> None:
        self.recovery_from = self.latent.req
        self._active_recovery_tau = float(cfg.RECOVERY_TAU * self.rng.uniform(0.55, 1.8))
        self._set_state(WorkloadState.RECOVERY)

    def set_workload_state(self, state: Optional[WorkloadState], request_rate: Optional[float]) -> str:
        if state is None and request_rate is None:
            raise ValueError("Provide workload_state and/or request_rate")
        if state is None:
            state = WorkloadState.CUSTOM
        if state == WorkloadState.CUSTOM and request_rate is None:
            raise ValueError("workload_state 'custom' requires request_rate")
        if state == WorkloadState.SPIKE:
            self.trigger_spike()
        elif state == WorkloadState.PEAK:
            self._set_state(WorkloadState.PEAK)
            self.peak_hold = None
        elif state == WorkloadState.RECOVERY:
            self.trigger_recovery()
        else:
            self._set_state(state)
        if request_rate is not None:
            self.manual_rate = float(request_rate)
        return f"workload_state={self.state.value}" + (
            f", request rate pinned at {self.manual_rate:.0f} req/min" if self.manual_rate else "")

    def _advance_state(self) -> None:
        self.state_ticks += 1

        # Autonomous workload generator for the live demo. Spikes are created
        # by the simulator itself, so the frontend never has to synthesize them.
        if (self.state == WorkloadState.NORMAL and self.auto_spike_enabled
                and self.manual_rate is None):
            self._normal_ticks_since_event += 1
            if self._normal_ticks_since_event >= self._next_auto_spike_tick:
                peak_rates = getattr(cfg, "AUTO_SPIKE_PEAK_RATES", [cfg.AUTO_SPIKE_PEAK_RATE])
                peak_rate = float(self.rng.choice(peak_rates)) * float(self.rng.uniform(0.82, 1.12))
                spike_duration = max(3, int(cfg.AUTO_SPIKE_DURATION_TICKS * self.rng.uniform(0.75, 1.35)))
                peak_hold = max(2, int(cfg.AUTO_SPIKE_PEAK_HOLD_TICKS * self.rng.uniform(0.65, 1.45)))
                self._auto_spike_index += 1
                self._next_auto_spike_tick = self._schedule_next_auto_spike()
                self.trigger_spike(
                    peak_rate=peak_rate,
                    spike_duration=spike_duration,
                    peak_hold=peak_hold,
                )
                return
        if self.state == WorkloadState.SPIKE and self.state_ticks > self.spike_duration:
            self._set_state(WorkloadState.PEAK)
            self.state_ticks = 1
        elif (self.state == WorkloadState.PEAK and self.peak_hold is not None
              and self.state_ticks > self.peak_hold):
            self.trigger_recovery()
            self.state_ticks = 1
        elif (self.state == WorkloadState.RECOVERY
              and abs(self.latent.req - self._normal_demand_level) < 0.03 * max(self._normal_demand_level, 1.0)):
            self._set_state(WorkloadState.NORMAL)
            self.state_ticks = 1

    def _target_request(self) -> float:
        """Request rate the workload is 'asking for' this tick (before smoothing)."""
        if self.manual_rate is not None:
            return self.manual_rate
        k, n = self.state_ticks, cfg.NORMAL_RATE
        if self.state == WorkloadState.SPIKE:      # fast exponential ramp to the peak
            ramp = 1 - math.exp(-k / max(1.0, self._active_spike_tau))
            # Curvature is sampled once per event, not independently each tick.
            # This gives each burst coherent but distinct dynamics instead of a noisy
            # copy of the same fixed exponential curve.
            return n + (self.peak_rate - n) * (ramp ** self._active_ramp_exponent) * (1 + self.rng.normal(0, 0.018))
        if self.state == WorkloadState.PEAK:       # irregular plateau with event-specific sag
            sag = 1 - self._active_peak_sag * clamp(k / max(1, self.peak_hold or 12), 0, 1)
            return self.peak_rate * sag * (1 + self.rng.normal(0, self._active_peak_wobble))
        if self.state == WorkloadState.RECOVERY:   # event-specific recovery rate with small demand noise
            target_normal = self._normal_demand_level
            base = target_normal + (self.recovery_from - target_normal) * math.exp(-k / max(1.0, self._active_recovery_tau))
            return base * (1 + self.rng.normal(0, 0.012))
        # Normal demand follows a bounded stochastic drift, not a fixed repeating waveform.
        if k >= self._next_demand_drift_tick:
            self._normal_demand_target = clamp(
                self._normal_demand_target * float(self.rng.lognormal(mean=0.0, sigma=0.065)),
                n * 0.62, n * 1.48,
            )
            self._next_demand_drift_tick = k + int(self.rng.integers(3, 10))
        self._normal_demand_level += 0.32 * (self._normal_demand_target - self._normal_demand_level)
        micro_variation = float(self.rng.normal(0.0, 0.028))
        return self._normal_demand_level * (1.0 + micro_variation)

    # -------------------------------------------------------------- one tick
    def _step(self, ts: datetime) -> TelemetrySample:
        c, s, rng = self.configuration, self.latent, self.rng
        self._advance_state()

        # 1-2. workload -> request rate (tracks target, never jumps instantly)
        target = self._target_request()
        s.req = clamp_metric("request_rate",
                             s.req + cfg.REQ_TRACKING * (target - s.req) + rng.normal(0, (0.006 + 0.004 * min(1.0, abs(target-s.req)/max(target,1))) * target))

        # 3. load factor rho = demand / capacity (capacity derates when too hot)
        cap = serving_capacity(c, s.temp)
        rho = s.req / cap

        # 4. utilization follows load, with inertia
        s.util = clamp_metric("gpu_utilization", s.util + cfg.UTIL_TRACKING *
                              (utilization_target(c, rho) - s.util) + rng.normal(0, 0.4))
        # 5. power follows utilization
        power = clamp_metric("gpu_power", power_from_util(c, s.util) + rng.normal(0, 2.5))
        # 6. memory: config-dominated, slow drift
        s.mem = clamp_metric("memory", s.mem + cfg.MEM_TRACKING *
                             (memory_target(c, rho) - s.mem) + rng.normal(0, 0.04))
        # 7. temperature: exponential smoothing of the power-driven target
        s.temp = clamp_metric("temperature", cfg.THERMAL_SMOOTHING * s.temp +
                              (1 - cfg.THERMAL_SMOOTHING) * temperature_target(power) + rng.normal(0, 0.08))
        # 8. latency follows queueing curve; throughput follows min(demand, capacity)
        s.lat = clamp_metric("latency", s.lat + cfg.LAT_TRACKING *
                             (latency_target(c, rho) - s.lat) + rng.normal(0, 1.0))
        s.thr = clamp_metric("throughput", s.thr + cfg.THR_TRACKING *
                             (throughput_target(s.req, cap) - s.thr))

        # observed values = latent + small measurement noise (does not feed back)
        req_o = clamp_metric("request_rate", s.req * (1 + rng.normal(0, 0.008)))
        thr_o = clamp_metric("throughput", min(s.thr * (1 + rng.normal(0, 0.008)), req_o, cap))
        util_o = clamp_metric("gpu_utilization", s.util + rng.normal(0, 0.3))
        mem_o = clamp_metric("memory", s.mem + rng.normal(0, 0.03))
        temp_o = clamp_metric("temperature", s.temp + rng.normal(0, 0.05))
        lat_o = clamp_metric("latency", s.lat * (1 + rng.normal(0, 0.012)))

        # Generate request-level latency samples for an empirical P95 estimate.
        # These are simulated requests, not measurements from real inference traffic.
        latency_p95_o = clamp_metric("latency", sample_p95_latency(lat_o, rho, rng))

        # 9. SLA: the configured latency limit applies to P95, not central latency.
        sla = check_sla(self.sla, latency_p95_o, thr_o, temp_o)

        # 10. store
        self.sample_counter += 1
        n = c.gpu_allocation * c.replicas
        sample = TelemetrySample(
            sample_id=self.sample_counter, timestamp=ts, workload_state=self.state,
            gpu_utilization=round(util_o, 1), gpu_power=round(power, 1),
            temperature=round(temp_o, 1), memory=round(mem_o, 2),
            request_rate=int(round(req_o)), latency=round(lat_o, 1),
            latency_p95_ms=round(latency_p95_o, 1), throughput=int(round(thr_o)),
            sla_status=sla.sla_status, latency_ok=sla.latency_ok,
            throughput_ok=sla.throughput_ok, temperature_ok=sla.temperature_ok,
            serving_capacity=int(round(cap)), num_gpus=n,
            total_power=round(power * n, 1), total_memory=round(mem_o * n, 2),
            configuration=c.model_copy())
        self.history.append(sample)
        return sample

    def tick(self) -> TelemetrySample:
        """One live tick, stamped with the real clock."""
        return self._step(_utc_now())

    def step(self, n: int = 1) -> List[TelemetrySample]:
        return [self.tick() for _ in range(n)]

    # --------------------------------------------------- initial 100 samples
    def _generate_initial_dataset(self) -> None:
        """Create a varied, workload-driven history with randomized event timing.

        There is deliberately no fixed spike at a hard-coded sample index. The
        automatic state machine may produce zero, one, or several events based
        on a per-run random schedule; explicit seeds remain reproducible.
        """
        end = _utc_now()
        start = end - timedelta(seconds=cfg.SAMPLE_INTERVAL_S * (cfg.INITIAL_SAMPLES - 1))
        # Let the first event occur at a different point in each fresh run.
        self._next_auto_spike_tick = self._schedule_next_auto_spike()
        for i in range(1, cfg.INITIAL_SAMPLES + 1):
            self._step(start + timedelta(seconds=cfg.SAMPLE_INTERVAL_S * (i - 1)))
        self.initial_dataset: List[TelemetrySample] = list(self.history)[:cfg.INITIAL_SAMPLES]
        # Manual/live spikes also auto-recover; no indefinite peak hold.

    # --------------------------------------------------------- configuration
    def apply_configuration(self, update: ConfigurationUpdate) -> Configuration:
        merged = {**self.configuration.model_dump(),
                  **update.model_dump(exclude_none=True)}
        new = Configuration(**merged)
        if new != self.configuration:
            recent = list(self.history)[-5:]
            self._before = self._avg(recent)
            self._before_config = self.configuration.model_copy()
            self._change_id = self.sample_counter
        self.configuration = new
        return new

    def estimate_steady_state(self, c: Configuration,
                              request_rate: Optional[float] = None) -> SteadyStateEstimate:
        """Noise-free what-if: where would this config settle at this request rate?"""
        req = clamp_metric("request_rate", request_rate if request_rate is not None else self.latent.req)
        temp = 60.0
        for _ in range(12):   # fixed point of capacity(temp) <-> power <-> temperature
            cap = serving_capacity(c, temp)
            rho = req / cap
            util = clamp_metric("gpu_utilization", utilization_target(c, rho))
            power = clamp_metric("gpu_power", power_from_util(c, util))
            temp = clamp_metric("temperature", temperature_target(power))
        lat = clamp_metric("latency", latency_target(c, rho))
        lat_p95 = clamp_metric("latency", estimate_p95_latency(lat, rho))
        thr = clamp_metric("throughput", min(throughput_target(req, cap), req))
        mem = clamp_metric("memory", memory_target(c, rho))
        n = c.gpu_allocation * c.replicas
        sla = check_sla(self.sla, lat_p95, thr, temp)
        return SteadyStateEstimate(
            configuration=c, request_rate=round(req, 1), gpu_utilization=round(util, 1),
            gpu_power=round(power, 1), temperature=round(temp, 1), memory=round(mem, 2),
            latency=round(lat, 1), latency_p95_ms=round(lat_p95, 1),
            throughput=round(thr, 1), serving_capacity=round(cap, 1),
            num_gpus=n, total_power=round(power * n, 1), total_memory=round(mem * n, 2),
            sla=sla, feasible=sla.sla_status == "PASS")

    # ----------------------------------------------------- before / after
    def _avg(self, samples: List[TelemetrySample]) -> Dict[str, float]:
        return {k: round(float(np.mean([getattr(s, k) for s in samples])), 2)
                for k in self.COMPARE_KEYS}

    def comparison(self):
        if self._before is None or self._change_id is None:
            return None, None, None, None
        after_samples = [s for s in self.history if s.sample_id > self._change_id][-5:]
        if not after_samples:
            return self._before, None, None, None
        after = self._avg(after_samples)
        delta = {k: round(after[k] - self._before[k], 2) for k in self.COMPARE_KEYS}
        return self._before, after, delta, after_samples[-1].sla_status

    # ------------------------------------------------------------- accessors
    def get_cluster(self):
        """Return the dynamic state of the five simulated GPUs."""
        return build_cluster(self)

    @property
    def latest(self) -> TelemetrySample:
        return self.history[-1]

    def get_history(self, include_live: bool = False, limit: Optional[int] = None) -> List[TelemetrySample]:
        data = list(self.history) if include_live else list(self.initial_dataset)
        return data[-limit:] if limit else data


# ============================================================ export helpers
def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def flatten(sample: TelemetrySample) -> Dict:
    d = sample.model_dump(mode="json")
    conf = d.pop("configuration")
    d.update({f"config_{k}": v for k, v in conf.items()})
    return d


def samples_to_csv(samples: List[TelemetrySample]) -> str:
    rows = [flatten(s) for s in samples]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def samples_to_json(samples: List[TelemetrySample]) -> str:
    return json.dumps({"data_source": cfg.DATA_SOURCE_LABEL, "count": len(samples),
                       "samples": [s.model_dump(mode="json") for s in samples]}, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export the initial 100 SIMULATED GPU TELEMETRY samples")
    ap.add_argument("--export-dir", default="data")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()
    sim = GPUSimulator(seed=a.seed)
    os.makedirs(a.export_dir, exist_ok=True)
    ds = sim.get_history()
    with open(os.path.join(a.export_dir, "gpupilot_initial_100.json"), "w") as f:
        f.write(samples_to_json(ds))
    with open(os.path.join(a.export_dir, "gpupilot_initial_100.csv"), "w", newline="") as f:
        f.write(samples_to_csv(ds))
    print(f"Wrote {len(ds)} samples ({cfg.DATA_SOURCE_LABEL}) to {a.export_dir}/")
