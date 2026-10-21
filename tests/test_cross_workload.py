from engine.cross_workload import DEFAULT_WORKLOADS, GPU_POOL, plan_cross_workload


def test_cross_workload_plan_assigns_each_gpu_at_most_once():
    result = plan_cross_workload()
    assigned_ids = [item["gpu_id"] for item in result["assignments"]]
    assert len(assigned_ids) == len(set(assigned_ids))
    assert result["allocated_gpu_count"] + result["idle_gpu_count"] == len(GPU_POOL)


def test_cross_workload_plan_handles_parallel_workload_types():
    result = plan_cross_workload()
    kinds = {item["kind"] for item in result["workloads"]}
    assert {"llm", "embedding", "image", "batch"} <= kinds
    assert result["workload_count"] == 4
    assert all("decision_reason" in item for item in result["workloads"])


def test_cross_workload_plan_respects_gpu_memory_compatibility():
    workloads = [{
        "id": "large-model", "kind": "llm", "demand_rpm": 300,
        "memory_gb": 90, "priority": 5, "max_latency_ms": 500,
        "min_throughput_rpm": 100,
    }]
    result = plan_cross_workload(workloads)
    assert result["workloads"][0]["assigned_gpus"] == []
    assert result["workloads"][0]["status"] == "QUEUED"


def test_cross_workload_plan_uses_unique_workload_ids():
    workloads = [
        {"id": "same", "kind": "llm", "demand_rpm": 100},
        {"id": "same", "kind": "embedding", "demand_rpm": 100},
    ]
    try:
        plan_cross_workload(workloads)
    except ValueError as exc:
        assert "Duplicate workload id" in str(exc)
    else:
        raise AssertionError("Expected duplicate workload IDs to be rejected")


def test_cross_workload_plan_rejects_unknown_workload_kind():
    try:
        plan_cross_workload([{"id": "bad", "kind": "video", "demand_rpm": 10}])
    except ValueError as exc:
        assert "Unsupported workload kind" in str(exc)
    else:
        raise AssertionError("Expected unsupported workload kind to be rejected")


def test_cross_workload_plan_is_explicitly_simulated():
    result = plan_cross_workload(DEFAULT_WORKLOADS)
    assert result["data_source"] == "SIMULATED CROSS-WORKLOAD PLAN"
    assert "not physical measurements" in result["disclaimer"]



def test_gpu_can_be_shared_by_multiple_compatible_workloads():
    gpus = [{"id": "GPU-A", "model": "Test GPU", "memory_gb": 80, "capacity_factor": 1.0, "power_w": 300}]
    workloads = [
        {"id": "priority-llm", "kind": "llm", "demand_rpm": 100, "priority": 5, "memory_gb": 20},
        {"id": "background-batch", "kind": "batch", "demand_rpm": 100, "priority": 1, "memory_gb": 10},
    ]
    result = plan_cross_workload(workloads, gpus)
    allocations = result["gpus"][0]["allocations"]
    assert len(allocations) == 2
    assert sum(a["share_pct"] for a in allocations) <= 100
    assert result["allocated_gpu_count"] == 1


def test_fractional_shares_never_exceed_gpu_capacity():
    result = plan_cross_workload()
    for gpu in result["gpus"]:
        assert gpu["allocated_share_pct"] <= 100
        assert gpu["remaining_memory_gb"] >= 0
        assert abs(gpu["allocated_share_pct"] + (100 - gpu["allocated_share_pct"]) - 100) < 1e-6


def test_each_gpu_allocation_references_a_known_workload():
    result = plan_cross_workload()
    workload_ids = {w["id"] for w in result["workloads"]}
    for gpu in result["gpus"]:
        for allocation in gpu["allocations"]:
            assert allocation["workload_id"] in workload_ids
