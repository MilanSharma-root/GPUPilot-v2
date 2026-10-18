"""GPUPilot FastAPI service.

SIMULATED GPU TELEMETRY: the default provider is the mathematical simulator,
not a physical NVIDIA GPU. The service exposes the original simulator APIs plus
prediction, sustainability, one-shot autopilot and continuous closed-loop control.
"""
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

import config as cfg
from engine.controller import GPUPilotController
from engine.optimizer import GPUPilotOptimizer
from engine.predictor import GPUPilotPredictor
from engine.sustainability import DEFAULT_ENV, compare_impact, estimate_impact
from models import (ComparisonResponse, Configuration, ConfigurationResponse,
                    ConfigurationUpdate, EvaluateRequest, HealthResponse,
                    HistoryResponse, SimulationStatus, SLAConfig, SpikeRequest,
                    SteadyStateEstimate, TelemetryControl, TelemetryControlResponse,
                    TelemetrySample)
from simulator import GPUSimulator, samples_to_csv, samples_to_json
from environment_monitor import initial_monitor_state, scan_environment

sim = GPUSimulator()
optimizer = GPUPilotOptimizer(sim)
predictor = GPUPilotPredictor()
controller = GPUPilotController(sim, optimizer, predictor)
environment_monitor_state = initial_monitor_state()


class OptimizationWorker:
    """Single background worker; dashboard polling never launches candidate searches."""
    def __init__(self):
        self.task: Optional[asyncio.Task] = None
        self.interval_s = 2.0
        self.running_job = False
        self.last_error = None

    @property
    def running(self):
        return self.task is not None and not self.task.done()

    async def _loop(self):
        while True:
            try:
                self.running_job = True
                await asyncio.to_thread(optimizer.optimize)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self.running_job = False
            await asyncio.sleep(self.interval_s)

    def start(self):
        if not self.running:
            self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.running:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None


optimization_worker = OptimizationWorker()


class Runner:
    """Background asyncio task that calls sim.tick() every interval."""
    def __init__(self) -> None:
        self.task: Optional[asyncio.Task] = None
        self.interval = cfg.SAMPLE_INTERVAL_S

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def _loop(self) -> None:
        while True:
            sim.tick()
            await asyncio.sleep(self.interval)

    def start(self, interval: Optional[float] = None) -> None:
        if interval:
            self.interval = interval
        if not self.running:
            self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.running:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None


runner = Runner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prime the cache once before serving requests, then refresh only in the worker.
    await asyncio.to_thread(optimizer.optimize)
    optimization_worker.start()
    yield
    await optimization_worker.stop()
    await controller.stop()
    await runner.stop()


app = FastAPI(
    title="GPUPilot - Autonomous AI Workload Sustainability Controller",
    description=(
        "Simulator-first AI workload controller. All default telemetry is "
        "SIMULATED GPU TELEMETRY, not from a physical NVIDIA GPU."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _status(message: str = "") -> SimulationStatus:
    return SimulationStatus(
        running=runner.running,
        tick_interval_s=runner.interval,
        workload_state=sim.state,
        samples_generated=sim.sample_counter,
        message=message,
    )


def _optimization_payload(result):
    return {
        "status": "success",
        "current_configuration": result.current_configuration.model_dump(),
        "recommended_configuration": result.recommended_configuration.model_dump(),
        "current_estimate": result.current_estimate.model_dump(),
        "recommended_estimate": result.recommended_estimate.model_dump(),
        "candidates_checked": result.candidates_checked,
        "feasible_candidates": result.feasible_candidates,
        "reason": result.reason,
        "optimizer_status": {"worker_running": optimization_worker.running, "evaluation_running": optimization_worker.running_job, "last_evaluation_at": optimizer.last_evaluated_at, "last_error": optimization_worker.last_error},
    }


# --------------------------------------------------- hardware/software monitor
@app.get("/api/environment/monitor")
async def get_environment_monitor():
    """Return simulated inventory and the latest TinyFish update scan."""
    environment_monitor_state["configured"] = bool(__import__("os").getenv("TINYFISH_API_KEY", "").strip())
    if not environment_monitor_state.get("last_scan_at") and not environment_monitor_state.get("findings"):
        environment_monitor_state["message"] = (
            "TinyFish API key is configured. Run a scan to check public release information."
            if environment_monitor_state["configured"] else
            "Set TINYFISH_API_KEY to enable live release searches. The inventory is simulated."
        )
    return environment_monitor_state


@app.post("/api/environment/scan")
async def scan_environment_updates():
    """Search current public release information with TinyFish; never modifies drivers."""
    global environment_monitor_state
    environment_monitor_state = await asyncio.to_thread(scan_environment)
    return environment_monitor_state


# ------------------------------------------------------------------- health
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", simulation_running=runner.running,
                          samples_generated=sim.sample_counter, workload_state=sim.state)


# ---------------------------------------------------------------- telemetry
@app.get("/api/telemetry", response_model=TelemetrySample)
async def get_telemetry(advance: bool = Query(False, description="Generate one new tick first")):
    return sim.tick() if advance else sim.latest


@app.post("/api/telemetry", response_model=TelemetryControlResponse)
async def set_workload(body: TelemetryControl):
    try:
        msg = sim.set_workload_state(body.workload_state, body.request_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TelemetryControlResponse(workload_state=sim.state, request_rate_override=sim.manual_rate,
                                    message=msg, telemetry=sim.latest)


@app.get("/api/history", response_model=HistoryResponse)
async def get_history(include_live: bool = False, limit: Optional[int] = Query(None, ge=1)):
    samples = sim.get_history(include_live=include_live, limit=limit)
    return HistoryResponse(count=len(samples), samples=samples)


@app.get("/api/history/export")
async def export_history(format: Literal["json", "csv"] = "csv", include_live: bool = False):
    samples = sim.get_history(include_live=include_live)
    if not samples:
        raise HTTPException(status_code=404, detail="No telemetry samples available")
    body, mt = (samples_to_csv(samples), "text/csv") if format == "csv" else (samples_to_json(samples), "application/json")
    return Response(content=body, media_type=mt,
                    headers={"Content-Disposition": f'attachment; filename="gpupilot_simulated_telemetry.{format}"'})


# ------------------------------------------------------------ configuration
@app.get("/api/configuration", response_model=Configuration)
async def get_configuration():
    return sim.configuration


@app.post("/api/configuration", response_model=ConfigurationResponse)
async def set_configuration(update: ConfigurationUpdate):
    previous = sim.configuration.model_copy()
    new = sim.apply_configuration(update)
    est = sim.estimate_steady_state(new)
    return ConfigurationResponse(previous_configuration=previous, configuration=new,
                                 expected_steady_state=est,
                                 message="Configuration applied. Telemetry will converge over the next several ticks.")


@app.post("/api/configuration/evaluate", response_model=SteadyStateEstimate)
async def evaluate_configuration(body: EvaluateRequest):
    return sim.estimate_steady_state(body.configuration, body.request_rate)


@app.get("/api/sla", response_model=SLAConfig)
async def get_sla():
    return sim.sla


@app.post("/api/sla", response_model=SLAConfig)
async def set_sla(sla: SLAConfig):
    sim.sla = sla
    return sim.sla


# --------------------------------------------------------------- simulation
@app.post("/api/simulation/start", response_model=SimulationStatus)
async def start(tick_interval_s: Optional[float] = Query(None, gt=0.05, le=60)):
    runner.start(tick_interval_s)
    # The demo is autonomous: starting telemetry also starts the closed-loop
    # controller so automatic spikes can trigger real configuration decisions.
    controller.start(2.0)
    return _status("Continuous simulation and autonomous GPUPilot controller running.")


@app.post("/api/simulation/stop", response_model=SimulationStatus)
async def stop():
    await runner.stop()
    await controller.stop()
    return _status("Continuous simulation and autonomous controller stopped.")


@app.post("/api/simulation/reset", response_model=SimulationStatus)
async def reset():
    was_running = runner.running
    was_auto = controller.running
    sim.reset()
    controller.last_cycle = None
    controller.events.clear()
    controller.last_action_at = None
    controller.last_gpu_selection = []
    if was_running:
        runner.start(runner.interval)
    if was_auto:
        controller.start(2.0)
    return _status("Simulation reset to initial state (100 samples regenerated); automatic spikes are enabled.")


@app.post("/api/simulation/spike", response_model=SimulationStatus)
async def spike(body: Optional[SpikeRequest] = None):
    body = body or SpikeRequest()
    sim.trigger_spike(body.peak_rate, body.spike_duration_ticks, body.peak_hold_ticks)
    # Track which generated samples belong to an explicit anomaly demo. Manual
    # workload spikes clear the marker so they do not raise anomaly alerts solely
    # because the UI button was clicked.
    sample_start = sim.sample_counter + 1
    sample_end = (
        sim.sample_counter + body.spike_duration_ticks
        + (body.peak_hold_ticks or cfg.DEFAULT_PEAK_HOLD_TICKS) + 12
    )
    if body.trigger_type == "anomaly":
        sim._gpupilot_anomaly_sample_start = sample_start
        sim._gpupilot_anomaly_sample_end = sample_end
        sim._gpupilot_manual_sample_start = None
        sim._gpupilot_manual_sample_end = None
    else:
        sim._gpupilot_anomaly_sample_start = None
        sim._gpupilot_anomaly_sample_end = None
        sim._gpupilot_manual_sample_start = sample_start
        sim._gpupilot_manual_sample_end = sample_end
    hint = "" if runner.running else " (simulation is stopped - call /api/simulation/start or /api/simulation/step)"
    alert_kind = "Anomaly scenario triggered" if body.trigger_type == "anomaly" else "Workload spike triggered"
    return _status(f"{alert_kind} toward {body.peak_rate:.0f} req/min.{hint}")


@app.post("/api/simulation/step", response_model=list[TelemetrySample])
async def step(n: int = Query(1, ge=1, le=500)):
    return sim.step(n)


# ------------------------------------------------------------- prediction
@app.get("/api/prediction")
async def prediction():
    return predictor.predict(sim.get_history(include_live=True, limit=300)).as_dict()


# ------------------------------------------------------------- optimization
@app.get("/api/optimize")
@app.get("/api/optimization")
async def optimize():
    try:
        result = await asyncio.to_thread(optimizer.optimize)
        return _optimization_payload(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


# ------------------------------------------------------------- sustainability
@app.get("/api/impact")
async def impact():
    result = optimizer.last_result or optimizer.optimize()
    return {
        "current": estimate_impact(result.current_estimate, DEFAULT_ENV),
        "recommended": estimate_impact(result.recommended_estimate, DEFAULT_ENV),
        "comparison": compare_impact(result.current_estimate, result.recommended_estimate, DEFAULT_ENV),
    }


# ----------------------------------------------------------- one-shot autopilot
@app.post("/api/autopilot/run")
@app.post("/api/autopilot")
async def autopilot():
    try:
        result = await asyncio.to_thread(optimizer.optimize)
        current = result.current_configuration
        recommended = result.recommended_configuration
        applied = current
        changed = recommended != current

        if changed and result.feasible_candidates > 0:
            update = ConfigurationUpdate(**recommended.model_dump())
            applied = sim.apply_configuration(update)

            # Observe the simulator after the change. If it is not running,
            # advance a few ticks manually so the one-shot demo still converges.
            if runner.running:
                await asyncio.sleep(max(1.0, min(3.0, runner.interval * 3)))
            else:
                sim.step(3)
            controller._event(
                "configuration_applied",
                "One-shot autopilot applied a lower-impact feasible configuration.",
                before=current.model_dump(),
                after=applied.model_dump(),
                power_improvement_pct=round(
                    ((result.current_estimate.total_power - result.recommended_estimate.total_power)
                     / result.current_estimate.total_power * 100.0)
                    if result.current_estimate.total_power else 0.0, 2
                ),
            )

        after_estimate = sim.estimate_steady_state(applied, sim.latest.request_rate)
        impact = compare_impact(result.current_estimate, after_estimate, DEFAULT_ENV)

        # Safety check against the actual post-change estimate.
        if changed and not after_estimate.feasible:
            rollback = ConfigurationUpdate(**current.model_dump())
            sim.apply_configuration(rollback)
            applied = current
            controller._event(
                "rollback",
                "Post-change SLA verification failed; the previous safe configuration was restored.",
                before=recommended.model_dump(),
                after=current.model_dump(),
            )
            return {
                "status": "rollback",
                "message": "GPUPilot rolled back because the post-change estimate failed the SLA.",
                "before": {"configuration": current.model_dump(), "estimate": result.current_estimate.model_dump()},
                "recommended": {"configuration": recommended.model_dump(), "estimate": result.recommended_estimate.model_dump()},
                "applied_configuration": applied.model_dump(),
                "post_change_estimate": after_estimate.model_dump(),
                "impact": impact,
            }

        return {
            "status": "success",
            "message": ("GPUPilot optimized the workload and automatically applied the recommended configuration."
                         if changed else "Current configuration is already the best feasible operating point."),
            "before": {"configuration": current.model_dump(), "estimate": result.current_estimate.model_dump()},
            "recommended": {"configuration": recommended.model_dump(), "estimate": result.recommended_estimate.model_dump()},
            "applied_configuration": applied.model_dump(),
            "post_change_estimate": after_estimate.model_dump(),
            "impact": impact,
            "optimization": {"candidates_checked": result.candidates_checked,
                              "feasible_candidates": result.feasible_candidates,
                              "reason": result.reason},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autopilot failed: {e}")


# ------------------------------------------------------- continuous autopilot
@app.post("/api/autopilot/start")
async def start_autopilot(interval_s: Optional[float] = Query(None, gt=1, le=60)):
    controller.start(interval_s)
    return {"status": "success", "message": "Continuous GPUPilot autopilot started.", "autopilot": controller.status()}


@app.post("/api/autopilot/stop")
async def stop_autopilot():
    await controller.stop()
    return {"status": "success", "message": "Continuous GPUPilot autopilot stopped.", "autopilot": controller.status()}


@app.get("/api/autopilot/status")
async def autopilot_status():
    return controller.status()




# ----------------------------------------------------------- precision quality gate
@app.post("/api/quality/evaluate")
async def evaluate_precision_quality(body: dict):
    """Clearly-labelled simulator-only precision quality gate; no real model is loaded."""
    from quality import evaluate_simulated_quality
    try:
        candidate = str(body.get("candidate_precision", "INT8")).upper()
        reference = str(body.get("reference_precision", "FP16")).upper()
        threshold = float(body.get("allowed_degradation_pct", 2.0))
        result = evaluate_simulated_quality(reference, candidate, threshold)
        return result
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/api/quality/status")
async def precision_quality_status():
    from quality import evaluate_simulated_quality
    return evaluate_simulated_quality("FP16", "INT8", 2.0)


# --------------------------------------------------------------- dashboard
@app.get("/api/cluster")
async def cluster_state():
    gpus = sim.get_cluster()
    active = [g for g in gpus if g["active"]]
    return {
        "gpu_count": len(gpus),
        "active_count": len(active),
        "idle_count": len(gpus) - len(active),
        "gpus": gpus,
        "workload_state": sim.state,
        "request_rate": sim.latest.request_rate,
        "data_source": cfg.DATA_SOURCE_LABEL,
    }

@app.get("/api/metrics/realtime")
async def realtime_metrics(limit: int = Query(60, ge=1, le=300)):
    samples = sim.get_history(include_live=True, limit=limit)
    return {"samples": [s.model_dump(mode="json") for s in samples]}

@app.get("/api/environmental-impact")
async def environmental_impact():
    result = optimizer.last_result or optimizer.optimize()
    current = estimate_impact(result.current_estimate, DEFAULT_ENV)
    recommended = estimate_impact(result.recommended_estimate, DEFAULT_ENV)
    return {"current": current, "recommended": recommended, "comparison": compare_impact(result.current_estimate, result.recommended_estimate, DEFAULT_ENV)}

@app.get("/api/optimization/status")
async def optimization_status():
    result = optimizer.last_result or optimizer.optimize()
    return _optimization_payload(result)

@app.get("/api/optimization/activity")
async def optimization_activity():
    return {"events": controller.events[-50:]}

@app.get("/api/controller/status")
async def controller_status():
    return controller.status()

@app.get("/api/workloads")
async def workloads():
    t = sim.latest
    gpus = sim.get_cluster()
    active = [g["id"] for g in gpus if g["active"]]
    return {"workloads": [{
        "id": "workload-01",
        "name": "LLM Inference",
        "model": "Llama 3 70B",
        "requests_per_min": t.request_rate,
        "latency_ms": t.latency,
        "latency_p95_ms": t.latency_p95_ms,
        "throughput_req_min": t.throughput,
        "assigned_gpus": active,
        "status": "SLA Protected" if t.sla_status == "PASS" else "Scaling",
    }]}


def _anomaly_report(samples=None):
    """Detect anomalies from telemetry deviations, not from UI flags.

    A demand burst is labelled anomalous only when request-rate growth is
    accompanied by impact on utilization, power, or latency, or when an SLA
    limit is breached. Baselines use the latest 15 NORMAL-state observations.
    """
    samples = list(samples if samples is not None else sim.get_history(include_live=True, limit=600))
    if not samples:
        return {"status": "NORMAL", "detected": False, "severity": "none",
                "reason": "Waiting for telemetry samples.", "metrics": [], "sample_ids": [],
                "latest_sample_id": None, "baseline": {}}

    normal_samples = []
    anomalous_ids = []
    event_details = {}
    for sample in samples:
        baseline_samples = normal_samples[-15:]
        if len(baseline_samples) >= 5:
            baseline = {
                "request_rate": float(np.median([x.request_rate for x in baseline_samples])),
                "gpu_utilization": float(np.median([x.gpu_utilization for x in baseline_samples])),
                "gpu_power": float(np.median([x.gpu_power for x in baseline_samples])),
                "latency": float(np.median([x.latency for x in baseline_samples])),
                "temperature": float(np.median([x.temperature for x in baseline_samples])),
            }
            demand_jump = sample.request_rate >= max(baseline["request_rate"] * 1.22,
                                                      baseline["request_rate"] + 120)
            utilization_impact = sample.gpu_utilization >= baseline["gpu_utilization"] + 7
            power_impact = sample.gpu_power >= baseline["gpu_power"] * 1.07
            latency_impact = sample.latency >= baseline["latency"] * 1.18
            # A pre-existing SLA miss is not automatically an anomaly; the metric
            # must also deviate materially from its normal baseline.
            hard_limit = ((not sample.latency_ok and latency_impact) or
                          (not sample.temperature_ok and sample.temperature >= baseline["temperature"] + 4))
            state_value = getattr(sample.workload_state, "value", sample.workload_state)
            # Only the dedicated anomaly button's sample window is force-detected.
            # A manual spike remains a workload event unless telemetry independently
            # crosses the normal correlated-deviation / SLA thresholds.
            anomaly_start = getattr(sim, "_gpupilot_anomaly_sample_start", None)
            anomaly_end = getattr(sim, "_gpupilot_anomaly_sample_end", None)
            is_anomaly_demo_sample = (
                anomaly_start is not None and anomaly_end is not None
                and anomaly_start <= sample.sample_id <= anomaly_end
            )
            manual_start = getattr(sim, "_gpupilot_manual_sample_start", None)
            manual_end = getattr(sim, "_gpupilot_manual_sample_end", None)
            is_manual_spike_sample = (
                manual_start is not None and manual_end is not None
                and manual_start <= sample.sample_id <= manual_end
            )
            intentional_spike = (
                is_anomaly_demo_sample and state_value in ("spike", "peak", "recovery")
                and demand_jump
            )
            correlated_anomaly = (
                demand_jump and (utilization_impact or power_impact or latency_impact)
            )
            if (correlated_anomaly and not is_manual_spike_sample) or (hard_limit and not is_manual_spike_sample) or intentional_spike:
                anomalous_ids.append(sample.sample_id)
                metrics = []
                if demand_jump: metrics.append("request rate")
                if utilization_impact: metrics.append("GPU utilization")
                if power_impact: metrics.append("GPU power")
                if latency_impact: metrics.append("latency")
                if not sample.temperature_ok: metrics.append("temperature SLA")
                if not sample.latency_ok: metrics.append("latency SLA")
                if intentional_spike and not any(m in metrics for m in ("GPU utilization", "GPU power", "latency")):
                    metrics.append("workload spike")
                event_details[sample.sample_id] = {
                    "sample_id": sample.sample_id,
                    "timestamp": sample.timestamp.isoformat(),
                    "metrics": metrics,
                    "reason": ("Intentional workload spike detected against the normal request-rate baseline: " + ", ".join(metrics)
                               if intentional_spike else
                               "Request burst caused correlated telemetry deviation: " + ", ".join(metrics)
                               if demand_jump else "Configured SLA/thermal limit was breached: " + ", ".join(metrics)),
                    "severity": "critical" if hard_limit or not sample.throughput_ok else "warning",
                }
        if getattr(sample.workload_state, "value", sample.workload_state) == "normal":
            normal_samples.append(sample)

    latest = samples[-1]
    latest_id = latest.sample_id
    recent_anomaly = next((event_details[sid] for sid in reversed(anomalous_ids)
                           if latest_id - sid <= 6), None)
    if latest_id in event_details:
        status = "DETECTED"
        active = event_details[latest_id]
    elif recent_anomaly and getattr(latest.workload_state, "value", latest.workload_state) in ("peak", "recovery"):
        status = "RECOVERING"
        active = recent_anomaly
    else:
        status = "NORMAL"
        active = None

    current_baseline_samples = normal_samples[-15:]
    current_baseline = {}
    if current_baseline_samples:
        current_baseline = {
            "request_rate": round(float(np.median([x.request_rate for x in current_baseline_samples])), 1),
            "gpu_utilization": round(float(np.median([x.gpu_utilization for x in current_baseline_samples])), 1),
            "gpu_power": round(float(np.median([x.gpu_power for x in current_baseline_samples])), 1),
            "latency": round(float(np.median([x.latency for x in current_baseline_samples])), 1),
        }
    return {
        "status": status,
        "detected": status == "DETECTED",
        "severity": (active or {}).get("severity", "none"),
        "reason": (active or {}).get("reason", "Telemetry is within the learned normal range."),
        "metrics": (active or {}).get("metrics", []),
        "sample_ids": anomalous_ids[-100:],
        "latest_sample_id": latest_id,
        "last_detected_at": (event_details[anomalous_ids[-1]]["timestamp"] if anomalous_ids else None),
        "baseline": current_baseline,
    }


def _dashboard_events(limit: int = 24):
    """Build graph/dashboard events from real simulator state transitions and controller events."""
    samples = sim.get_history(include_live=True, limit=min(cfg.HISTORY_MAXLEN, 600))
    events = []
    previous = None
    for sample in samples:
        state = sample.workload_state.value
        if previous is not None and state != previous:
            label = {
                "spike": ("traffic_spike", "Traffic Spike", "Workload demand started rising."),
                "peak": ("peak", "Peak Load", "Workload reached the peak phase."),
                "recovery": ("recovery", "Recovery", "Workload entered automatic recovery."),
                "normal": ("normal", "Normal", "Workload returned to normal."),
            }.get(state)
            if label:
                event_type, title, message = label
                events.append({
                    "timestamp": sample.timestamp.isoformat(),
                    "type": event_type,
                    "title": title,
                    "message": message,
                    "sample_id": sample.sample_id,
                })
        previous = state

    for item in controller.events:
        event = dict(item)
        event["type"] = event.get("event", "controller")
        event["title"] = str(event.get("event", "Controller event")).replace("_", " ").title()
        events.append(event)

    anomaly = _anomaly_report(samples)
    anomaly_ids = set(anomaly.get("sample_ids", []))
    previous_anomalous = False
    for sample in samples:
        is_anomalous = sample.sample_id in anomaly_ids
        if is_anomalous and not previous_anomalous:
            events.append({
                "timestamp": sample.timestamp.isoformat(),
                "type": "anomaly_detected",
                "title": "Telemetry Anomaly Detected",
                "message": "Telemetry departed from the recent normal baseline across correlated metrics.",
                "sample_id": sample.sample_id,
            })
        previous_anomalous = is_anomalous

    events.sort(key=lambda x: x.get("timestamp", ""))
    return events[-limit:]


def _optimization_reasons(result):
    """Explain the actual optimizer decision in human-readable, data-derived terms."""
    t = sim.latest
    current = result.current_estimate
    recommended = result.recommended_estimate
    reasons = []

    if t.gpu_power >= 275:
        reasons.append("High GPU power consumption detected.")
    elif t.gpu_power >= 240:
        reasons.append("GPU power is elevated for the current workload.")

    if t.latency_p95_ms < sim.sla.max_latency_ms * 0.85:
        reasons.append("P95 latency has sufficient headroom under the configured SLA.")
    elif t.latency_p95_ms < sim.sla.max_latency_ms:
        reasons.append("P95 latency remains within the configured SLA.")
    else:
        reasons.append("P95 latency exceeds the configured SLA limit; prioritize tail-latency recovery.")

    if current.throughput > 0 and recommended.throughput >= current.throughput * 0.98:
        reasons.append(f"{recommended.configuration.precision} maintains throughput within the current SLA envelope.")

    if recommended.configuration.batch_size > current.configuration.batch_size:
        reasons.append("Larger batch size improves serving efficiency for the current demand.")

    if recommended.configuration.gpu_allocation > current.configuration.gpu_allocation:
        reasons.append("Additional simulated GPU capacity is required for the current workload.")
    elif recommended.configuration.gpu_allocation < current.configuration.gpu_allocation:
        reasons.append("Unnecessary simulated GPU capacity can be released during this workload.")

    if recommended.configuration.precision != current.configuration.precision:
        reasons.append(f"Precision change {current.configuration.precision} → {recommended.configuration.precision} reduces estimated impact.")

    if not reasons:
        reasons.append("Current configuration is already the best feasible operating point for the current workload.")

    return reasons[:4]


@app.get("/api/dashboard")
async def dashboard():
    """Consolidated, live dashboard payload for the judge-facing UI."""
    result = optimizer.last_result or optimizer.optimize()
    current_impact = estimate_impact(result.current_estimate, DEFAULT_ENV)
    recommended_impact = estimate_impact(result.recommended_estimate, DEFAULT_ENV)
    comparison = compare_impact(result.current_estimate, result.recommended_estimate, DEFAULT_ENV)
    t = sim.latest
    gpus = sim.get_cluster()
    history = sim.get_history(include_live=True, limit=3600)
    controller_status = controller.status()
    configuration_history = []
    seen_configs = set()
    for sample in reversed(history):
        c = sample.configuration
        key = (c.gpu_allocation, c.batch_size, c.precision, c.replicas, c.concurrency)
        if key in seen_configs:
            continue
        seen_configs.add(key)
        configuration_history.append({
            "timestamp": sample.timestamp.isoformat(),
            "workload_state": sample.workload_state.value,
            "request_rate": round(float(sample.request_rate), 1),
            "configuration": c.model_dump(),
        })
        if len(configuration_history) >= 8:
            break
    configuration_history.reverse()
    power_saved = max(0.0, float(comparison["changes"]["power_saved_w"]))
    throughput = max(float(result.recommended_estimate.throughput), 0.0)
    recommended_power = max(float(result.recommended_estimate.total_power), 0.001)
    sustainability = {
        "energy_saved_kwh_per_hour": round(power_saved / 1000.0, 4),
        "estimated_water_reduction_l_per_hour": round(max(0.0, float(comparison["changes"]["water_reduction_l_per_hour"])), 4),
        "thermal_reduction_c": round(max(0.0, float(comparison["changes"]["temperature_reduction_c"])), 2),
        "energy_efficiency_req_per_w": round(throughput / recommended_power, 3),
        "estimated_water_l_per_hour": round(float(recommended_impact["estimated_water_l_per_hour"]), 4),
        "pue_assumption": DEFAULT_ENV.pue,
        "wue_assumption_l_per_kwh": DEFAULT_ENV.wue_l_per_kwh,
    }
    return {
        "data_source": cfg.DATA_SOURCE_LABEL,
        "telemetry": t.model_dump(mode="json"),
        "history": [s.model_dump(mode="json") for s in history],
        "anomaly": _anomaly_report(history),
        "cluster": {
            "gpu_count": len(gpus),
            "active_count": sum(1 for g in gpus if g["active"]),
            "idle_count": sum(1 for g in gpus if not g["active"]),
            "gpus": gpus,
            "workload_state": sim.state.value,
            "required_for_demand": optimizer.required_gpu_count(t.request_rate),
            "capacity_gap": max(0, optimizer.required_gpu_count(t.request_rate) - (sim.configuration.gpu_allocation * sim.configuration.replicas)),
        },
        "configuration": sim.configuration.model_dump(),
        "optimization": {
            **_optimization_payload(result),
            "reasons": _optimization_reasons(result),
            "expected_impact": comparison,
        },
        "sla": {
            "configuration": sim.sla.model_dump(),
            "status": t.sla_status,
            "latency_ok": t.latency_ok,
            "p95_latency_ms": t.latency_p95_ms,
            "throughput_ok": t.throughput_ok,
            "temperature_ok": t.temperature_ok,
        },
        "sustainability": sustainability,
        "configuration_history": configuration_history,
        "simulation": {
            "running": runner.running,
            "tick_interval_s": runner.interval,
            "workload_state": sim.state.value,
            "samples_generated": sim.sample_counter,
        },
        "autonomous_mode": {
            "enabled": controller.running,
            "interval_s": controller.interval_s,
            "last_cycle": controller.last_cycle,
        },
        "optimizer_status": {"worker_running": optimization_worker.running, "evaluation_running": optimization_worker.running_job, "last_evaluation_at": optimizer.last_evaluated_at, "last_error": optimization_worker.last_error},
        "recent_events": _dashboard_events(),
        "workloads": [{
            "id": "workload-01",
            "name": "LLM Inference",
            "model": "Llama 3 70B",
            "requests_per_min": t.request_rate,
            "latency_ms": t.latency,
            "throughput_req_min": t.throughput,
            "assigned_gpus": [g["id"] for g in gpus if g["active"]],
            "status": "SLA Protected" if t.sla_status == "PASS" else "SLA Failed",
        }],
    }

# --------------------------------------------------------------------- demo
@app.get("/api/demo/scenario")
async def demo_scenario():
    """Run a deterministic short demo sequence without requiring a physical GPU."""
    sim.reset()
    sim.trigger_spike(peak_rate=1150, spike_duration=8, peak_hold=6)
    samples = sim.step(14)
    return {
        "status": "success",
        "scenario": "normal_to_spike",
        "samples": [s.model_dump(mode="json") for s in samples],
        "latest": sim.latest.model_dump(mode="json"),
        "message": "Demo spike generated. Run /api/autopilot/run to optimize and apply a new configuration.",
    }


@app.get("/api/demo/comparison", response_model=ComparisonResponse)
async def comparison():
    before, after, delta, status = sim.comparison()
    note = ("No configuration change yet." if before is None else
            "Waiting for new samples after the change." if after is None else
            "Averages of up to 5 samples each; negative delta = reduction.")
    return ComparisonResponse(configuration_before=sim._before_config,
                              configuration_after=sim.configuration, before=before,
                              after=after, delta=delta, after_sla_status=status, note=note)


@app.get("/api/autopilot/events")
async def autopilot_events():
    return {"events": controller.events[-50:]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
