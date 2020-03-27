#!/bin/sh
rosrun xacro xacro -o crawler.urdf crawler.urdf.xacro
check_urdf crawler.urdf
t=$?
echo "$t"
if [ $t ]
then
    echo "correctly checked the URDF"
    conda activate ./envs
    python3 sim_crawler_experiment.py
fi
