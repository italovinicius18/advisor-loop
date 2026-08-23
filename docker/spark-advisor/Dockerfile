FROM apache/spark:3.5.0
USER root
RUN pip install --no-cache-dir boto3 delta-spark==3.2.0
# jars Delta p/ o Spark (delta-spark instala os py; os jars vem via
# spark.jars.packages=io.delta:delta-spark_2.12:3.2.0 nas confs fixas do
# launcher em runtime - ver advisor_loop/backends/launcher.py)
COPY spark_advisor/driver.py /opt/advisor/driver.py
COPY queries/ /opt/advisor/queries/
ENV PYTHONPATH=/opt/advisor
USER spark
