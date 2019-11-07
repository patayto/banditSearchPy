#!/bin/bash
OUTPUT_DIR_JOB_1=/user/${USER}/minf/job1
OUTPUT_DIR_JOB_2=/user/${USER}/minf/job2

# Hadoop won't start if the output directory already exists
hdfs dfs -rm -r $OUTPUT_DIR_JOB_1
hdfs dfs -rm -r $OUTPUT_DIR_JOB_2

hadoop jar /opt/hadoop/hadoop-2.9.2/share/hadoop/tools/lib/hadoop-streaming-2.9.2.jar \
    -D mapreduce.job.name=${USER}_minf_job1 \
    -input dummy.txt \
    -output $OUTPUT_DIR_JOB_1 \
    -mapper 'Map1.py 2 0.05' \
    -reducer 'Reduce1.py 2 2' \
    -file Map1.py \
    -file Reduce1.py
