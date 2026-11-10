FROM apache/spark:4.2.0-scala2.13-java17-python3-ubuntu
USER root
# APENAS boto3 no pip: NAO instalar pyspark/delta-spark aqui, senao o pyspark do
# pip sombreia /opt/spark/jars e o Delta some do classpath do driver.
RUN pip install --no-cache-dir boto3
# JARs assados na imagem (evita resolucao Ivy em runtime, que falha no pod por
# cache nao-gravavel + necessidade de egress ao Maven Central):
#  - Delta 4.4.0 (Scala 2.13) — par oficial do Spark 4.2.0
#  - S3A: hadoop-aws 3.5.0 (Hadoop que o Spark 4.2 empacota) + AWS SDK v2 bundle
# Versoes descobertas a partir dos POMs (hadoop-project 3.5.0 -> awssdk 2.35.4).
ARG DELTA_VER=4.4.0
ARG HADOOP_VER=3.5.0
ARG AWS_SDK_VER=2.35.4
# hadoop-aws 3.5.0 depende (obrigatorio, via audit + stream) do S3A Analytics
# Accelerator; versao casada no hadoop-project 3.5.0.
ARG AAL_VER=1.3.1
ARG MVN=https://repo1.maven.org/maven2
RUN cd /opt/spark/jars && \
    curl -fsSLO ${MVN}/io/delta/delta-spark_2.13/${DELTA_VER}/delta-spark_2.13-${DELTA_VER}.jar && \
    curl -fsSLO ${MVN}/io/delta/delta-storage/${DELTA_VER}/delta-storage-${DELTA_VER}.jar && \
    curl -fsSLO ${MVN}/org/apache/hadoop/hadoop-aws/${HADOOP_VER}/hadoop-aws-${HADOOP_VER}.jar && \
    curl -fsSLO ${MVN}/software/amazon/awssdk/bundle/${AWS_SDK_VER}/bundle-${AWS_SDK_VER}.jar && \
    curl -fsSLO ${MVN}/software/amazon/s3/analyticsaccelerator/analyticsaccelerator-s3/${AAL_VER}/analyticsaccelerator-s3-${AAL_VER}.jar
COPY spark_advisor/driver.py /opt/advisor/driver.py
COPY queries/ /opt/advisor/queries/
COPY queries-tpch/ /opt/advisor/queries-tpch/
# Pod template (nodeAffinity ao pool de egress) assado na imagem: o driver le o
# executor.podTemplateFile do proprio FS e o launcher le o driver.podTemplateFile
# do seu FS -> todos usam esta imagem, entao o arquivo esta presente nos tres.
COPY docker/spark-advisor/pod-template-egress.yaml /opt/advisor/pod-template-egress.yaml
ENV PYTHONPATH=/opt/advisor
USER spark
