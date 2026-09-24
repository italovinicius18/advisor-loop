# Controller in-cluster do advisor-loop (padrao exp05-controller da dissertacao).
# Roda o orquestrador (propose->spark-submit->feedback) DENTRO do namespace, via
# kubectl local (ssh_cmd=[bash,-c] no datasets.yaml). Nao precisa de Spark: quem
# roda spark-submit e o pod launcher (imagem spark-advisor). So precisa de
# Python + deps do orquestrador + kubectl.
FROM python:3.12-slim

ARG KUBECTL_VER=v1.27.7
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates \
    && curl -fsSLo /usr/local/bin/kubectl \
        "https://dl.k8s.io/release/${KUBECTL_VER}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Deps do caminho de execucao (L0): sem pyspark/pandas/pyarrow. duckdb entra
# porque wiring importa probes.duckdb_probe no topo; boto3 para o upload.
RUN pip install --no-cache-dir pydantic httpx pyyaml python-dotenv duckdb boto3

WORKDIR /opt/advisor-loop
COPY advisor_loop/ advisor_loop/
COPY spark_advisor/ spark_advisor/
COPY packs/ packs/
COPY campaigns/ campaigns/
COPY queries/ queries/
COPY docker/advisor-controller/datasets.cluster.yaml datasets.yaml
COPY docker/advisor-controller/upload_results.py upload_results.py
COPY docker/advisor-controller/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/opt/advisor-loop
ENTRYPOINT ["/opt/advisor-loop/entrypoint.sh"]
