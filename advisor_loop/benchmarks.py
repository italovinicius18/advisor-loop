"""Nome do dataset -> benchmark e SF.

TPC-DS mantem os nomes legados 'sf<N>' (journals, campanhas e o datasets.yaml
ja usam). TPC-H usa 'tpch-sf<N>': mesmo SF, tabelas e queries proprias
(silver/sf<N>/tpch/delta/parquet, queries-tpch/). O driver Spark repete esta
regra em `spark_advisor.driver.resolve_dataset` (arquivo solto na imagem).
"""
from __future__ import annotations

import re

_RE = re.compile(r"^(?:(?P<bench>[a-z]+)-)?sf(?P<sf>\d+)$")
_QUERIES = {"tpcds": "queries", "tpch": "queries-tpch"}


def _parse(dataset: str) -> tuple[str, int]:
    m = _RE.match(str(dataset).lower())
    if not m:
        raise ValueError(f"dataset invalido: {dataset!r} (use 'sf<N>' ou 'tpch-sf<N>')")
    bench = m.group("bench") or "tpcds"
    if bench not in _QUERIES:
        raise ValueError(f"benchmark desconhecido em {dataset!r}: {bench!r}")
    return bench, int(m.group("sf"))


def benchmark_of(dataset: str) -> str:
    return _parse(dataset)[0]


def sf_of(dataset: str) -> int:
    return _parse(dataset)[1]


def sf_segment(dataset: str) -> str:
    """'tpch-sf500' -> 'sf500' (segmento do path na Silver)."""
    return f"sf{sf_of(dataset)}"


def queries_subdir(dataset: str) -> str:
    return _QUERIES[benchmark_of(dataset)]
