# Matriz experimental da dissertacao (oraculo da campanha Phase-2)

`training_data.parquet` e a matriz reparada da dissertacao (exp05): a fonte do
**oraculo** contra o qual a campanha Phase-2 e falsificada. Para cada
`(scale_factor, query)`, o menor `duration_ms` com `status='success'` e o
`t*` -- o melhor que o grid experimental achou. O advisor (LLM/random) tenta
igualar, aproximar ou bater esse `t*`.

## Proveniencia

- Origem: `data/consolidated/training_data.parquet` da branch de trabalho
  `worktree-banca-figuras` (identica em `worktree-exp05-rerun-matrix`), commit
  `01f7078`. Copiado (nao regenerado) em 2026-09-07.
- Conteudo: 68 celulas / ~20.196 execucoes; 55 colunas.
- Dados brutos por tras: `rerun-202608/` (47 celulas) + `rerun-202609/`
  (21 celulas) -- exatamente os `CLEAN_METRICS_DIRS` da dissertacao.
- **Nunca** derivado de `gold/` nem `legacy-pre-repair/` (dados pre-reparo).

## Colunas relevantes para o oraculo

`scale_factor`, `query`, `duration_ms`, `status`, `executor_cores`,
`executor_memory_gb`, `workers`. Scale factors presentes: 10, 50, 100, 250,
500, 750. A campanha usa SF100.

## Caveat de subespaco (spec secao 2.4)

A matriz varreu apenas `executor_cores x executor_memory_gb x workers` (AQE
sempre on; `shuffle_partitions`/`advisory_partition_mb`/`broadcast_threshold`/
`driver_memory` nunca variados). Logo `t*` e um **piso sobre um subespaco** --
o schema livre do advisor pode em principio ir abaixo dele. E um baseline
documentado, nao um limite absoluto.

## Uso

```bash
python -m analysis.report <journal.jsonl> data/matrix/training_data.parquet
```
