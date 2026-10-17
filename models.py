"""models.py - Pydantic request/response models. All data is SIMULATED GPU TELEMETRY."""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

import config as cfg

Precision = Literal["FP32", "FP16", "INT8"]
DATA_SOURCE = Literal["SIMULATED GPU TELEMETRY"]


class WorkloadState(str, Enum):
    NORMAL = "normal"
    SPIKE = "spike"
    PEAK = "peak"
    RECOVERY = "recovery"
    CUSTOM = "custom"   # request rate pinned manually via POST /api/telemetry


# ------------------------------------------------------------ configuration
class Configuration(BaseModel):
    gpu_allocation: int = Field(1, ge=1, le=8, description="GPUs per replica")
    batch_size: int = Field(16, ge=1, le=128)
    precision: Precision = "FP16"
    replicas: int = Field(1, ge=1, le=8)
    concurrency: int = Field(32, ge=1, le=256)

    @field_validator("precision", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.upper() if isinstance(v, str) else v


class ConfigurationUpdate(BaseModel):
    """Partial update - omitted fields keep their current value."""
    gpu_allocation: Optional[int] = Field(None, ge=1, le=8)
    batch_size: Optional[int] = Field(None, ge=1, le=128)
    precision: Optional[Precision] = None
    replicas: Optional[int] = Field(None, ge=1, le=8)
    concurrency: Optional[int] = Field(None, ge=1, le=256)

    @field_validator("precision", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.upper() if isinstance(v, str) else v


# --------------------------------------------------------------------- SLA
class SLAConfig(BaseModel):
    max_latency_ms: float = Field(200.0, gt=0)
    min_throughput_req_min: float = Field(700.0, gt=0)
    max_temperature_c: float = Field(85.0, gt=0)


class SLAResult(BaseModel):
    latency_ok: bool
    throughput_ok: bool
    temperature_ok: bool
    sla_status: Literal["PASS", "FAIL"]


# --------------------------------------------------------------- telemetry
class TelemetrySample(BaseModel):
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    sample_id: int
    timestamp: datetime
    workload_state: WorkloadState
    gpu_utilization: float          # %   (per-GPU average)
    gpu_power: float                # W   (per-GPU average)
    temperature: float              # C   (per-GPU average)
    memory: float                   # GB  (per-GPU average)
    request_rate: int               # req/min offered
    latency: float                  # ms
    throughput: int                 # req/min served
    sla_status: Literal["PASS", "FAIL"]
    latency_ok: bool
    throughput_ok: bool
    temperature_ok: bool
    # extras for the optimizer / dashboard
    serving_capacity: int           # req/min the current config can serve
    num_gpus: int                   # gpu_allocation * replicas
    total_power: float              # W, summed over all allocated GPUs
    total_memory: float             # GB, summed over all allocated GPUs
    configuration: Configuration


class HistoryResponse(BaseModel):
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    count: int
    samples: List[TelemetrySample]


class TelemetryControl(BaseModel):
    workload_state: Optional[WorkloadState] = None
    request_rate: Optional[float] = Field(None, ge=300, le=2200,
                                          description="Pin the target request rate (req/min)")


class TelemetryControlResponse(BaseModel):
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    workload_state: WorkloadState
    request_rate_override: Optional[float]
    message: str
    telemetry: TelemetrySample


class SpikeRequest(BaseModel):
    peak_rate: float = Field(cfg.DEFAULT_PEAK_RATE, ge=600, le=1500)
    spike_duration_ticks: int = Field(cfg.DEFAULT_SPIKE_TICKS, ge=3, le=200)
    peak_hold_ticks: Optional[int] = Field(
        None, ge=1, description="Ticks to hold the peak before auto-recovery. "
        "null = hold until recovery is requested (best for live demos).")
    trigger_type: Literal["manual", "anomaly"] = Field(
        "manual", description="manual creates a workload alert; anomaly runs the anomaly-alert demo.")


# ----------------------------------------------------------- configuration API
class SteadyStateEstimate(BaseModel):
    """Noise-free steady-state what-if estimate (SIMULATED)."""
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    configuration: Configuration
    request_rate: float
    gpu_utilization: float
    gpu_power: float
    temperature: float
    memory: float
    latency: float
    throughput: float
    serving_capacity: float
    num_gpus: int
    total_power: float
    total_memory: float
    sla: SLAResult
    feasible: bool


class ConfigurationResponse(BaseModel):
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    previous_configuration: Configuration
    configuration: Configuration
    expected_steady_state: SteadyStateEstimate
    message: str


class EvaluateRequest(BaseModel):
    configuration: Configuration
    request_rate: Optional[float] = Field(None, ge=300, le=2200)


# --------------------------------------------------------------- misc
class SimulationStatus(BaseModel):
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    running: bool
    tick_interval_s: float
    workload_state: WorkloadState
    samples_generated: int
    message: str = ""


class HealthResponse(BaseModel):
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    status: str
    simulation_running: bool
    samples_generated: int
    workload_state: WorkloadState


class ComparisonResponse(BaseModel):
    """BEFORE vs AFTER the most recent configuration change."""
    data_source: DATA_SOURCE = cfg.DATA_SOURCE_LABEL
    configuration_before: Optional[Configuration]
    configuration_after: Configuration
    before: Optional[Dict[str, float]]
    after: Optional[Dict[str, float]]
    delta: Optional[Dict[str, float]]
    after_sla_status: Optional[str]
    note: str
