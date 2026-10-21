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
