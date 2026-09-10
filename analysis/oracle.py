"""Oraculo da matriz experimental (dissertacao, exp05).

O baseline falsificavel da campanha Phase-2: para cada (scale_factor, query),
o menor duration_ms entre as execucoes com status='success' na matriz reparada.
E o "melhor que o grid achou" -- o LLM/random tentam igualar, aproximar ou bater.

CAVEAT (spec secao 2.4): a matriz varreu apenas executor_cores x
executor_memory_gb x workers (AQE sempre on; shuffle_partitions/advisory/
broadcast/driver_memory nunca variados). t* e portanto um piso sobre um
subespaco -- o schema livre pode ir abaixo. Documentado, nao um bug.

Fonte de dados: SOMENTE dirs limpos (rerun-202608/202609 -> training_data.parquet
consolidado). NUNCA gold/legacy-pre-repair.
"""
from __future__ import annotations
import duckdb


def oracle_table(parquet_path: str) -> dict[tuple[int, str], float]:
    """min(duration_ms) onde status='success', agrupado por (scale_factor, query).

    Retorna {(sf:int, query:str): t_star_ms:float}."""
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT CAST(scale_factor AS INTEGER) AS sf,
                   query,
                   MIN(CAST(duration_ms AS DOUBLE)) AS t_star
            FROM read_parquet(?)
            WHERE status = 'success' AND duration_ms IS NOT NULL
            GROUP BY sf, query
            """,
            [parquet_path],
        ).fetchall()
    finally:
        con.close()
    return {(int(sf), str(q)): float(t) for sf, q, t in rows}


def oracle_table_median(parquet_path: str) -> dict[tuple[int, str], float]:
    """Regua JUSTA: min sobre as celulas (cores, mem, workers) da MEDIANA por
    celula das execucoes success. `oracle_table` (min puro) e o melhor de ~45
    amostras (15 celulas x 3 tries) e favorece a amostra mais sortuda; o LLM
    reporta a mediana de k tries por config. Comparar mediana com mediana
    remove esse vies (no SF100 e de 1-6%: a matriz e estavel -- documentado,
    nao o que explica o gap).

    Retorna {(sf:int, query:str): t_star_median_ms:float}."""
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT sf, query, MIN(med) AS t_star_med
            FROM (
                SELECT CAST(scale_factor AS INTEGER) AS sf, query,
                       executor_cores, executor_memory_gb, workers,
                       MEDIAN(CAST(duration_ms AS DOUBLE)) AS med
                FROM read_parquet(?)
                WHERE status = 'success' AND duration_ms IS NOT NULL
                GROUP BY 1, 2, 3, 4, 5
            )
            GROUP BY sf, query
            """,
            [parquet_path],
        ).fetchall()
    finally:
        con.close()
    return {(int(sf), str(q)): float(t) for sf, q, t in rows}


def best_config(parquet_path: str, sf: int, query: str) -> dict:
    """Linha vencedora (argmin duration_ms, status='success') para (sf, query).

    Retorna a linha inteira como dict (cores/mem/workers/duration + o que houver)."""
    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            WHERE status = 'success' AND duration_ms IS NOT NULL
              AND CAST(scale_factor AS INTEGER) = ? AND query = ?
            ORDER BY CAST(duration_ms AS DOUBLE) ASC
            LIMIT 1
            """,
            [parquet_path, int(sf), str(query)],
        )
        cols = [d[0] for d in row.description]
        vals = row.fetchone()
    finally:
        con.close()
    if vals is None:
        raise KeyError(f"sem execucao success para sf={sf} query={query}")
    return dict(zip(cols, vals))
