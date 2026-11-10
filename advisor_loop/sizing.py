"""v4 -- harness SCALE-AWARE: envelope de viabilidade dimensionado pelos DADOS.

Diagnostico (multi-SF, 2026-09-24): tres pecas eram cegas a escala --
baseline fixa (8c x 1w x 8g nao termina em sf>=500 -> loop sem perfil da
iteracao 0), random uniforme nas faixas (em SF grande sorteia timeouts de
200 min) e timeout absoluto. Tudo aqui deriva de UMA medida feita antes de
qualquer execucao Spark e valida para qualquer benchmark: o volume do
snapshot Delta ATUAL das tabelas que a query referencia (soma dos add.size
do _delta_log, nao `mc du` -- sf500 tem 812 GiB de objetos por versoes nao
vacuumadas contra ~420 GiB reais).

Envelope (constantes calibradas por principio e VALIDADAS offline contra as
menores celulas que completaram na matriz -- a matriz nunca entra no loop):
  baseline  : mem total alvo = max(16, 0.25 x data_gb); cores totais alvo =
              max(8, data_gb / 8); executores de 4c com 2 GB/core (3 GB/core
              acima de 150 GB) -> a menor config "razoavel" dentro do envelope.
  guardian  : piso duro (rejeita antes de gastar horas): mem total >= 0.10 x
              data_gb, cores totais >= data_gb / 32, mem/executor >= 8 GB se
              data_gb >= 5. `data_gb` e o volume POR QUERY (tabelas que ela
              toca; medido: sf500 q03 78 / q23 173 / q50 88 GB). Calibrado para
              ACEITAR toda menor celula que completou na matriz e REJEITAR o
              default 8c x 1w x 8g onde ele estourou 200 min (sf500 q23/q50,
              sf>=750); em sf250 q23 (90 GB) o default completou em 27 min e
              fica na borda (rejeitado: 9 GB de piso) -- custo zero.
  mem/core  : 2 GB (<100 GB), 3 GB (100-300 GB), 4 GB (>=300 GB) -> a baseline
              de q23@sf1000 (369 GB) sai 4c x 12w x 16g = 192 GB, a menor
              celula que completou la na matriz.
  timeout   : teto por proposta = max(3 x baseline, 20 min) (ver loop.py).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from advisor_loop.episode import ConfigVector

GB = 1024 ** 3


# ---------------------------------------------------------------- medicao
def tables_in_sql(sql: str, known: list[str]) -> list[str]:
    """Tabelas de `known` referenciadas no SQL (match por palavra inteira)."""
    return [t for t in known if re.search(rf"\b{re.escape(t)}\b", sql)]


def delta_snapshot_bytes(s3, bucket: str, prefix: str) -> int:
    """Tamanho do snapshot ATUAL de uma tabela Delta = soma dos `add.size`
    vivos apos aplicar todos os commits JSON de _delta_log em ordem (adds e
    removes). Sem checkpoint (as Silver tem 1-2 commits); se houver
    `_last_checkpoint`, os commits anteriores ao checkpoint podem ter sido
    limpos -> levanta para o chamador cair no fallback (listar parquet)."""
    prefix = prefix.rstrip("/") + "/_delta_log/"
    keys = []
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []) or []:
            keys.append(o["Key"])
        token = resp.get("NextContinuationToken")
        if not token:
            break
    if any(k.endswith("_last_checkpoint") for k in keys):
        raise NotImplementedError("tabela com checkpoint Delta: usar fallback de listagem")
    commits = sorted(k for k in keys if re.search(r"/\d{20}\.json$", k))
    live: dict[str, int] = {}
    for k in commits:
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read().decode()
        for line in body.splitlines():
            if not line.strip():
                continue
            act = json.loads(line)
            if "add" in act:
                live[act["add"]["path"]] = int(act["add"].get("size") or 0)
            elif "remove" in act:
                live.pop(act["remove"]["path"], None)
    return sum(live.values())


def parquet_listing_bytes(s3, bucket: str, prefix: str) -> int:
    """Fallback grosseiro: soma dos .parquet sob o prefixo (inclui versoes
    antigas -> superestima). So quando o log tem checkpoint."""
    total, token = 0, None
    prefix = prefix.rstrip("/") + "/"
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []) or []:
            if o["Key"].endswith(".parquet") and "_delta_log" not in o["Key"]:
                total += int(o.get("Size") or 0)
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return total


def _split_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def query_data_gb(s3, tables: dict[str, str], sql: str) -> tuple[float, dict[str, float]]:
    """(data_gb total, {tabela: gb}) das tabelas referenciadas pela query."""
    per: dict[str, float] = {}
    for t in tables_in_sql(sql, sorted(tables)):
        bucket, key = _split_s3(tables[t])
        try:
            b = delta_snapshot_bytes(s3, bucket, key)
        except NotImplementedError:
            b = parquet_listing_bytes(s3, bucket, key)
        per[t] = b / GB
    return round(sum(per.values()), 2), per


# ---------------------------------------------------------------- envelope
@dataclass
class Envelope:
    data_gb: float
    tables_gb: dict = field(default_factory=dict)
    # alvos da baseline
    target_mem_gb: float = 16
    target_cores: int = 8
    mem_per_core_gb: int = 2
    # pisos duros (Guardian)
    floor_mem_gb: float = 0
    floor_cores: int = 1
    floor_exec_mem_gb: int = 4

    def check(self, cfg: ConfigVector) -> tuple[bool, str]:
        tot_mem = cfg.executor_memory_gb * cfg.executor_instances
        tot_cores = cfg.executor_cores * cfg.executor_instances
        if tot_mem < self.floor_mem_gb:
            return False, (f"escala: memoria total {tot_mem}g < piso {self.floor_mem_gb:.0f}g "
                           f"para {self.data_gb:.0f}GB de dados")
        if tot_cores < self.floor_cores:
            return False, f"escala: cores totais {tot_cores} < piso {self.floor_cores}"
        if cfg.executor_memory_gb < self.floor_exec_mem_gb:
            return False, (f"escala: {cfg.executor_memory_gb}g por executor < piso "
                           f"{self.floor_exec_mem_gb}g (shuffle de {self.data_gb:.0f}GB de dados)")
        return True, "ok"

    def baseline(self, pinned: dict, caps: dict) -> ConfigVector:
        """Menor config razoavel dentro do envelope: executores de 4c."""
        cpe = 4
        mem_exec = max(self.floor_exec_mem_gb, cpe * self.mem_per_core_gb)
        inst = max(2, math.ceil(self.target_cores / cpe), math.ceil(self.target_mem_gb / mem_exec))
        inst = min(inst, 20, caps["max_pods"] - 1, caps["total_cores"] // cpe)
        while inst > 2 and mem_exec * inst + pinned.get("driver_memory_gb", 8) > caps["total_memory_gb"]:
            inst -= 1
        return ConfigVector(executor_cores=cpe, executor_memory_gb=mem_exec,
                            executor_instances=inst, **pinned)

    def render(self) -> str:
        top = ", ".join(f"{t}={g:.1f}GB" for t, g in sorted(self.tables_gb.items(),
                                                            key=lambda kv: -kv[1])[:4])
        return (f"## Envelope de escala (medido nos dados, antes de executar)\n"
                f"- dados tocados pela query: {self.data_gb:.1f} GB ({top})\n"
                f"- pisos de viabilidade (o Guardian REJEITA abaixo): memoria total >= "
                f"{self.floor_mem_gb:.0f} GB, cores totais >= {self.floor_cores}, "
                f"memoria por executor >= {self.floor_exec_mem_gb} GB\n"
                f"- ponto de partida sugerido: ~{self.target_cores} cores totais, "
                f"~{self.target_mem_gb:.0f} GB de memoria total, {self.mem_per_core_gb} GB/core")


def make_envelope(data_gb: float, tables_gb: dict | None = None) -> Envelope:
    mpc = 4 if data_gb >= 300 else (3 if data_gb >= 100 else 2)
    return Envelope(
        data_gb=data_gb, tables_gb=tables_gb or {},
        target_mem_gb=max(16.0, 0.25 * data_gb),
        target_cores=max(8, int(math.ceil(data_gb / 8))),
        mem_per_core_gb=mpc,
        floor_mem_gb=0.10 * data_gb,
        floor_cores=max(1, int(math.ceil(data_gb / 32))),
        floor_exec_mem_gb=8 if data_gb >= 5 else 4,
    )


class Sizer:
    """deps.sizer(ep) -> Envelope | None. Le o SQL da query e mede as tabelas
    no S3/MinIO; qualquer falha -> None (o loop volta ao comportamento v3)."""

    def __init__(self, s3, tables_by_dataset: dict[str, dict[str, str]], queries_dir):
        self.s3, self.tables_by_dataset, self.queries_dir = s3, tables_by_dataset, queries_dir
        self._cache: dict[tuple[str, str], Envelope | None] = {}

    def __call__(self, ep) -> Envelope | None:
        key = (ep.dataset, ep.queries[0])
        if key in self._cache:
            return self._cache[key]
        env = None
        try:
            tables = self.tables_by_dataset[ep.dataset]
            # queries_dir = dir TPC-DS; outros benchmarks ficam ao lado
            # (queries-tpch/), escolhidos pelo nome do dataset.
            from advisor_loop.benchmarks import queries_subdir
            sub = queries_subdir(ep.dataset)
            qdir = self.queries_dir if sub == "queries" else self.queries_dir.parent / sub
            sql = (qdir / f"{ep.queries[0]}.sql").read_text()
            gb, per = query_data_gb(self.s3, tables, sql)
            env = make_envelope(gb, per)
        except Exception as e:  # noqa: BLE001 -- degradacao graciosa p/ v3
            print(f"[sizing] sem envelope para {key}: {e}", flush=True)
        self._cache[key] = env
        return env
