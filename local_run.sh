#!/bin/bash
OUTPUT_DIR_JOB_1=job1
OUTPUT_DIR_JOB_2=job2

# Hadoop won't start if the output directory already exists
$HADOOP_HOME/bin/hdfs dfs -rm -r $OUTPUT_DIR_JOB_1
$HADOOP_HOME/bin/hdfs dfs -rm -r $OUTPUT_DIR_JOB_2

$HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.9.2.jar \
    -D mapreduce.job.name=minf_job1 \
    -input dummy.txt \
    -output $OUTPUT_DIR_JOB_1 \
    -mapper 'Map1.py 2 0.05' \
    -reducer 'Reduce1.py 2 2' \
    -file Map1.py \
    -file Reduce1.py
