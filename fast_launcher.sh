#!/bin/sh
rosrun xacro xacro -o crawler.urdf crawler.urdf.xacro
check_urdf crawler.urdf
t=$?
echo "$t"
if [ $t ]
then
    echo "correctly checked the URDF"
    source ./bin/activate
    python3 sim_crawler.py
fi