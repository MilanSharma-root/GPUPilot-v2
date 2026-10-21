"""Cross-workload allocation engine for GPUPilot's simulated cluster.

This module is intentionally simulator-only. It creates explainable, deterministic
allocation plans across heterogeneous workloads without claiming physical GPU control.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

GPU_POOL = [
    {"id": "GPU 01", "model": "RTX 4090", "memory_gb": 24, "capacity_factor": 1.00, "power_w": 450},
    {"id": "GPU 02", "model": "A100", "memory_gb": 80, "capacity_factor": 1.30, "power_w": 400},
    {"id": "GPU 03", "model": "A100", "memory_gb": 80, "capacity_factor": 1.30, "power_w": 400},
    {"id": "GPU 04", "model": "L40S", "memory_gb": 48, "capacity_factor": 1.10, "power_w": 350},
    {"id": "GPU 05", "model": "H100", "memory_gb": 80, "capacity_factor": 1.65, "power_w": 700},
]

WORKLOAD_PROFILES = {
    "llm": {"capacity": 420, "memory_gb": 32, "latency_weight": 1.35, "power_weight": 1.0},
    "embedding": {"capacity": 1500, "memory_gb": 8, "latency_weight": 1.10, "power_weight": 0.65},
    "image": {"capacity": 180, "memory_gb": 20, "latency_weight": 0.95, "power_weight": 1.1},
    "batch": {"capacity": 900, "memory_gb": 10, "latency_weight": 0.45, "power_weight": 0.55},
}

DEFAULT_WORKLOADS = [
    {"id": "llm-api", "name": "LLM Chat API", "kind": "llm", "model": "Llama 3 Instruct", "demand_rpm": 620, "priority": 5, "max_latency_ms": 220, "min_throughput_rpm": 500},
    {"id": "embedding-api", "name": "Embedding API", "kind": "embedding", "model": "text-embedding", "demand_rpm": 720, "priority": 3, "max_latency_ms": 160, "min_throughput_rpm": 600},
    {"id": "image-gen", "name": "Image Generation", "kind": "image", "model": "Diffusion XL", "demand_rpm": 120, "priority": 2, "max_latency_ms": 1800, "min_throughput_rpm": 90},
    {"id": "offline-batch", "name": "Batch Inference", "kind": "batch", "model": "Classifier Batch", "demand_rpm": 550, "priority": 1, "max_latency_ms": 5000, "min_throughput_rpm": 300},
]


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def plan_cross_workload(
    workloads: Iterable[Dict[str, Any]] | None = None,
    gpu_pool: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Greedily assign each simulated GPU to the workload with greatest urgency.

    Urgency combines declared priority, projected SLO pressure, and unmet demand.
    A GPU is assigned to at most one workload; memory fit and remaining demand are
    checked before assignment. All metrics in the response are simulated estimates.
    """
    raw_workloads = list(workloads if workloads is not None else DEFAULT_WORKLOADS)
    raw_gpus = [dict(g) for g in (gpu_pool if gpu_pool is not None else GPU_POOL)]
    if not raw_workloads:
        raise ValueError("At least one workload is required.")
    if not raw_gpus:
        raise ValueError("At least one simulated GPU is required.")

    normalized = []
    seen_ids = set()
    for index, raw in enumerate(raw_workloads):
        wid = str(raw.get("id") or f"workload-{index + 1}").strip()
        if wid in seen_ids:
            raise ValueError(f"Duplicate workload id: {wid}")
        seen_ids.add(wid)
        kind = str(raw.get("kind", "llm")).lower()
        if kind not in WORKLOAD_PROFILES:
            raise ValueError(f"Unsupported workload kind '{kind}'. Choose from {', '.join(WORKLOAD_PROFILES)}.")
        profile = WORKLOAD_PROFILES[kind]
        demand = _number(raw.get("demand_rpm"), 100, 0, 100000)
        priority = int(_number(raw.get("priority"), 3, 1, 5))
        normalized.append({
            **raw, "id": wid, "name": str(raw.get("name") or wid),
            "kind": kind, "model": str(raw.get("model") or kind),
            "demand_rpm": demand, "priority": priority,
            "max_latency_ms": _number(raw.get("max_latency_ms"), 500, 1, 60000),
            "min_throughput_rpm": _number(raw.get("min_throughput_rpm"), 0, 0, 100000),
            "memory_gb": _number(raw.get("memory_gb"), profile["memory_gb"], 1, 1000),
            "_profile": profile, "_gpus": [], "_capacity": 0.0,
        })

    # Highest priority first, then highest SLO urgency and demand pressure.
    def urgency(w: Dict[str, Any]) -> float:
        demand_ratio = w["demand_rpm"] / max(w["_profile"]["capacity"], 1)
        latency_tightness = 500.0 / w["max_latency_ms"]
        return w["priority"] * 100 + min(demand_ratio, 5) * 20 * w["_profile"]["latency_weight"] + latency_tightness

    normalized.sort(key=urgency, reverse=True)
    free_gpus = list(raw_gpus)
    assignments = []

    # Assign one compatible GPU at a time; continue while the workload is short
    # of its offered demand or below its declared minimum throughput.
    for workload in normalized:
        profile = workload["_profile"]
        while free_gpus:
            projected_capacity = workload["_capacity"]
            projected_latency = 35 + 260 * (workload["demand_rpm"] / max(projected_capacity, 1)) ** 2
            unmet_demand = workload["demand_rpm"] > projected_capacity
            latency_risk = projected_latency > workload["max_latency_ms"]
            throughput_shortfall = projected_capacity < workload["min_throughput_rpm"]
            if workload["_gpus"] and not (unmet_demand or latency_risk or throughput_shortfall):
                break
            compatible = [
                gpu for gpu in free_gpus
                if float(gpu.get("memory_gb", 0)) >= workload["memory_gb"]
            ]
            if not compatible:
                break
            # Pick a GPU that best balances capacity and estimated power for this workload.
            compatible.sort(key=lambda g: (
                -(profile["capacity"] * float(g.get("capacity_factor", 1.0)))
                + profile["power_weight"] * float(g.get("power_w", 0)) / 5
            ))
            gpu = compatible[0]
            free_gpus.remove(gpu)
            workload["_gpus"].append(gpu)
            workload["_capacity"] += profile["capacity"] * float(gpu.get("capacity_factor", 1.0))
            assignments.append({"gpu_id": gpu["id"], "gpu_model": gpu["model"], "workload_id": workload["id"], "workload_name": workload["name"]})

    result_workloads = []
    total_served = 0.0
    total_demand = 0.0
    total_power = 0.0
    for workload in normalized:
        profile = workload["_profile"]
        demand = workload["demand_rpm"]
        capacity = workload["_capacity"]
        served = min(demand, capacity)
        utilization = min(99.0, 18 + 78 * demand / max(capacity, 1)) if workload["_gpus"] else 0.0
        latency = 35 + 260 * (demand / max(capacity, 1)) ** 2 if workload["_gpus"] else 9999.0
        sla_ok = bool(workload["_gpus"]) and latency <= workload["max_latency_ms"] and served >= workload["min_throughput_rpm"]
        gpu_ids = [gpu["id"] for gpu in workload["_gpus"]]
        power = sum(float(gpu.get("power_w", 0)) * min(1.0, utilization / 100) for gpu in workload["_gpus"])
        if not gpu_ids:
            reason = "No compatible GPU capacity remained after higher-urgency workloads were allocated."
            status = "QUEUED"
        elif not sla_ok:
            reason = "Allocated available compatible capacity, but the simulated latency or throughput SLO is not met."
            status = "SLO AT RISK"
        else:
            reason = "Allocated compatible GPU capacity based on priority, demand, and SLO pressure."
            status = "RUNNING"
        result_workloads.append({
            "id": workload["id"], "name": workload["name"], "kind": workload["kind"],
            "model": workload["model"], "priority": workload["priority"],
            "demand_rpm": round(demand, 1), "estimated_capacity_rpm": round(capacity, 1),
            "estimated_throughput_rpm": round(served, 1), "estimated_latency_ms": round(latency, 1),
            "max_latency_ms": round(workload["max_latency_ms"], 1),
            "min_throughput_rpm": round(workload["min_throughput_rpm"], 1),
            "assigned_gpus": gpu_ids, "gpu_count": len(gpu_ids),
            "estimated_utilization_pct": round(utilization, 1),
            "estimated_power_w": round(power, 1), "sla_status": "PASS" if sla_ok else "FAIL",
            "status": status, "decision_reason": reason,
        })
        total_served += served
        total_demand += demand
        total_power += power

    used_ids = {a["gpu_id"] for a in assignments}
    return {
        "data_source": "SIMULATED CROSS-WORKLOAD PLAN",
        "policy": "priority + SLO pressure + demand + memory compatibility",
        "gpu_count": len(raw_gpus), "allocated_gpu_count": len(used_ids),
        "idle_gpu_count": len(raw_gpus) - len(used_ids),
        "workload_count": len(normalized), "queued_workload_count": sum(w["status"] == "QUEUED" for w in result_workloads),
        "sla_at_risk_count": sum(w["sla_status"] == "FAIL" for w in result_workloads),
        "total_demand_rpm": round(total_demand, 1), "estimated_served_rpm": round(total_served, 1),
        "estimated_cluster_power_w": round(total_power, 1),
        "assignments": assignments, "workloads": result_workloads,
        "gpus": [{
            **gpu,
            "assigned_workload_id": next((a["workload_id"] for a in assignments if a["gpu_id"] == gpu["id"]), None),
            "assigned_workload_name": next((a["workload_name"] for a in assignments if a["gpu_id"] == gpu["id"]), None),
            "status": "ALLOCATED" if gpu["id"] in used_ids else "IDLE",
        } for gpu in raw_gpus],
        "disclaimer": "All workload telemetry, capacity, latency, throughput, and power values are simulated estimates, not physical measurements.",
    }
