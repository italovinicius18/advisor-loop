# In-cluster DuckDB probe service. Minimal image: stdlib http.server + duckdb
# only (no FastAPI/uvicorn) so it builds/pulls fast next to MinIO in italo-unb.
FROM python:3.11-slim

RUN pip install --no-cache-dir "duckdb>=1.0"

# Pre-install the httpfs/delta extensions at build time so the running
# container needs no internet access (the cluster is namespace-scoped and
# may not have egress to DuckDB's extension CDN at pod start).
RUN python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL httpfs; INSTALL delta')"

WORKDIR /app
COPY service/ /app/

EXPOSE 8000

CMD ["python", "/app/duckdb_probe_server.py"]
