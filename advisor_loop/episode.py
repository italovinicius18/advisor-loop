import hashlib, json
from typing import Literal
from pydantic import BaseModel, Field

class ConfigVector(BaseModel):
    executor_cores: int = Field(ge=1, le=24)
    executor_memory_gb: int = Field(ge=4, le=96)
    executor_instances: int = Field(ge=1, le=20)
    driver_memory_gb: int = Field(ge=4, le=16)
    shuffle_partitions: int = Field(ge=100, le=4000)
    advisory_partition_mb: int = Field(ge=16, le=256)
    aqe_enabled: bool
    broadcast_threshold_mb: int = Field(ge=-1, le=512)

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

def task_key(workload: str, dataset: str, query: str, config_hash: str, try_n: int) -> str:
    raw = f"{workload}|{dataset}|{query}|{config_hash}|{try_n}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def episode_key(workload: str, dataset: str, query: str, try_n: int) -> str:
    """Chave de retomada independente de config_hash: um LLM real nao repete a
    mesma configuracao entre execucoes, entao a chave de resume nao pode
    depender da recomendacao gerada."""
    raw = f"{workload}|{dataset}|{query}|{try_n}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class QueryResult(BaseModel):
    status: Literal["success", "failed", "timeout", "killed"]
    duration_ms: float | None = None
    error: str | None = None

class Outcome(BaseModel):
    status: Literal["completed", "driver_crash", "aborted_by_supervisor",
                    "invalid_recommendation", "infeasible", "error"]
    per_query: dict[str, QueryResult] = {}
    wall_clock_s: float = 0.0
    cost_usd: float = 0.0

class Recommendation(BaseModel):
    config: ConfigVector
    rationale: str
    samples: int = 1

class Decision(BaseModel):
    action: Literal["continue", "kill"]
    reason: str
    evidence: list[str]
    confidence: float = Field(ge=0, le=1)

class Episode(BaseModel):
    workload: str
    dataset: str            # ex.: "sf100"
    queries: list[str]
    objective: Literal["min_time", "feasibility"] = "min_time"
    split: str = "none"     # ex.: "loso-scale:sf100"
    level: Literal["L0", "L1", "L2"] = "L0"
    provider: str           # model string pinada
