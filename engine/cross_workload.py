"""Simulator-only cross-workload GPU arbitration with fractional GPU sharing.

No physical GPU is controlled. Capacity, latency, throughput and power are modeled
estimates intended for reproducible prototype benchmarks.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable

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
    {"id":"llm-api","name":"LLM Chat API","kind":"llm","model":"Llama 3 Instruct","demand_rpm":620,"priority":5,"max_latency_ms":220,"min_throughput_rpm":500},
    {"id":"embedding-api","name":"Embedding API","kind":"embedding","model":"text-embedding","demand_rpm":720,"priority":3,"max_latency_ms":160,"min_throughput_rpm":600},
    {"id":"image-gen","name":"Image Generation","kind":"image","model":"Diffusion XL","demand_rpm":120,"priority":2,"max_latency_ms":1800,"min_throughput_rpm":90},
    {"id":"offline-batch","name":"Batch Inference","kind":"batch","model":"Classifier Batch","demand_rpm":550,"priority":1,"max_latency_ms":5000,"min_throughput_rpm":300},
]

def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try: number = float(value)
    except (TypeError, ValueError): number = default
    return max(minimum, min(maximum, number))

def plan_cross_workload(workloads: Iterable[Dict[str, Any]] | None = None,
                         gpu_pool: Iterable[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Plan fractional shares across GPUs; a GPU may host multiple compatible workloads.

    Each workload reserves its declared memory on each assigned GPU. GPU compute
    share is capped at 100%; allocations are never physically applied.
    """
    raw_workloads=list(workloads if workloads is not None else DEFAULT_WORKLOADS)
    gpus=[dict(g) for g in (gpu_pool if gpu_pool is not None else GPU_POOL)]
    if not raw_workloads: raise ValueError("At least one workload is required.")
    if not gpus: raise ValueError("At least one simulated GPU is required.")
    gpu_state=[]
    for g in gpus:
        g["memory_gb"]=_number(g.get("memory_gb"),0,0,10000)
        g["capacity_factor"]=_number(g.get("capacity_factor"),1,.01,100)
        g["power_w"]=_number(g.get("power_w"),0,0,100000)
        gpu_state.append({**g,"_remaining_share":1.0,"_remaining_memory":g["memory_gb"],"_allocations":[]})
    items=[]; seen=set()
    for i,raw in enumerate(raw_workloads):
        wid=str(raw.get("id") or f"workload-{i+1}").strip()
        if wid in seen: raise ValueError(f"Duplicate workload id: {wid}")
        seen.add(wid)
        kind=str(raw.get("kind","llm")).lower()
        if kind not in WORKLOAD_PROFILES: raise ValueError(f"Unsupported workload kind '{kind}'. Choose from {', '.join(WORKLOAD_PROFILES)}.")
        profile=WORKLOAD_PROFILES[kind]
        items.append({**raw,"id":wid,"name":str(raw.get("name") or wid),"kind":kind,
            "model":str(raw.get("model") or kind),"demand_rpm":_number(raw.get("demand_rpm"),100,0,100000),
            "priority":int(_number(raw.get("priority"),3,1,5)),
            "max_latency_ms":_number(raw.get("max_latency_ms"),500,1,60000),
            "min_throughput_rpm":_number(raw.get("min_throughput_rpm"),0,0,100000),
            "memory_gb":_number(raw.get("memory_gb"),profile["memory_gb"],1,10000),
            "_profile":profile,"_capacity":0.0,"_allocations":[]})
    def urgency(w):
        ratio=w["demand_rpm"]/max(w["_profile"]["capacity"],1)
        return w["priority"]*100+min(ratio,5)*20*w["_profile"]["latency_weight"]+500/w["max_latency_ms"]
    items.sort(key=urgency,reverse=True)
    allocation_records=[]
    for w in items:
        profile=w["_profile"]
        # Fill high-urgency demand first, preferring GPUs with best capacity per watt.
        while w["_capacity"] < max(w["demand_rpm"],w["min_throughput_rpm"]):
            candidates=[g for g in gpu_state if g["_remaining_share"] >= .10 and g["_remaining_memory"] >= w["memory_gb"]]
            if not candidates: break
            candidates.sort(key=lambda g: -(profile["capacity"]*g["capacity_factor"]/max(g["power_w"],1)))
            g=candidates[0]
            need=max(w["demand_rpm"],w["min_throughput_rpm"])-w["_capacity"]
            full_capacity=profile["capacity"]*g["capacity_factor"]
            share=min(g["_remaining_share"],max(.10,need/max(full_capacity,1)))
            share=min(share,g["_remaining_share"])
            if share < .10: break
            g["_remaining_share"]-=share
            g["_remaining_memory"]-=w["memory_gb"]
            cap=full_capacity*share
            w["_capacity"]+=cap
            record={"gpu_id":g["id"],"gpu_model":g["model"],"workload_id":w["id"],"workload_name":w["name"],
                    "share_pct":round(share*100,1),"memory_reserved_gb":w["memory_gb"],"estimated_capacity_rpm":round(cap,1)}
            g["_allocations"].append(record); w["_allocations"].append(record); allocation_records.append(record)
            if share >= .999: break
    results=[]; total_demand=total_served=total_power=0.0
    for w in items:
        demand=w["demand_rpm"]; capacity=w["_capacity"]; served=min(demand,capacity)
        # Simplified queueing proxy: latency rises quadratically as demand approaches capacity.
        latency=35+260*(demand/max(capacity,1))**2 if capacity>0 else 9999.0
        sla=bool(w["_allocations"]) and latency<=w["max_latency_ms"] and served>=w["min_throughput_rpm"]
        gpu_ids=list(dict.fromkeys(a["gpu_id"] for a in w["_allocations"]))
        util=min(99.0,18+78*demand/max(capacity,1)) if gpu_ids else 0.0
        power=sum(g["power_w"]*sum(a["share_pct"]/100 for a in g["_allocations"] if a["workload_id"]==w["id"])*min(1.0,util/100) for g in gpu_state)
        if not gpu_ids: status="QUEUED"; reason="No compatible GPU memory/share capacity remained after higher-urgency workloads."
        elif not sla: status="SLO AT RISK"; reason="Allocated a fractional GPU share, but simulated latency or minimum throughput SLO is not met."
        else: status="RUNNING"; reason="Fractional GPU share allocated using priority, demand, SLO pressure, memory fit, and estimated capacity per watt."
        results.append({"id":w["id"],"name":w["name"],"kind":w["kind"],"model":w["model"],"priority":w["priority"],
            "demand_rpm":round(demand,1),"estimated_capacity_rpm":round(capacity,1),"estimated_throughput_rpm":round(served,1),
            "estimated_latency_ms":round(latency,1),"max_latency_ms":round(w["max_latency_ms"],1),
            "min_throughput_rpm":round(w["min_throughput_rpm"],1),"assigned_gpus":gpu_ids,
            "gpu_shares":[{"gpu_id":a["gpu_id"],"share_pct":a["share_pct"]} for a in w["_allocations"]],
            "gpu_count":len(gpu_ids),"estimated_utilization_pct":round(util,1),"estimated_power_w":round(power,1),
            "memory_gb_per_gpu":w["memory_gb"],"sla_status":"PASS" if sla else "FAIL","status":status,"decision_reason":reason})
        total_demand+=demand; total_served+=served; total_power+=power
    gpus_out=[]
    for g in gpu_state:
        gpus_out.append({k:g[k] for k in ("id","model","memory_gb","capacity_factor","power_w")} | {
            "allocations":[{"workload_id":a["workload_id"],"workload_name":a["workload_name"],"share_pct":a["share_pct"],"memory_reserved_gb":a["memory_reserved_gb"]} for a in g["_allocations"]],
            "assigned_workload_id":g["_allocations"][0]["workload_id"] if len(g["_allocations"])==1 else None,
            "assigned_workload_name":g["_allocations"][0]["workload_name"] if len(g["_allocations"])==1 else ("Multiple workloads" if g["_allocations"] else None),
            "allocated_share_pct":round((1-g["_remaining_share"])*100,1),
            "remaining_memory_gb":round(g["_remaining_memory"],1),
            "status":"ALLOCATED" if g["_allocations"] else "IDLE"})
    used=sum(1 for g in gpus_out if g["status"]=="ALLOCATED")
    return {"data_source":"SIMULATED CROSS-WORKLOAD PLAN","policy":"priority + SLO pressure + demand + memory compatibility + fractional GPU sharing",
        "gpu_count":len(gpus),"allocated_gpu_count":used,"idle_gpu_count":len(gpus)-used,
        "workload_count":len(items),"queued_workload_count":sum(w["status"]=="QUEUED" for w in results),
        "sla_at_risk_count":sum(w["sla_status"]=="FAIL" for w in results),"total_demand_rpm":round(total_demand,1),
        "estimated_served_rpm":round(total_served,1),"estimated_cluster_power_w":round(total_power,1),
        "assignments":allocation_records,"workloads":results,"gpus":gpus_out,
        "disclaimer":"All workload telemetry, capacity, latency, throughput, and power values are simulated estimates, not physical measurements."}
