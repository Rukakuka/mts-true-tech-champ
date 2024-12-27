#!/bin/bash

sudo apt install python3-pip python3-colcon-common-extensions git -y && \
python3 -m pip install pymongo && \
pip install tornado && \
mkdir -p ws && \
cd ws && \
rm -rf src/ && \
mkdir -p src && \
cd src && \
git clone https://github.com/RobotWebTools/rosbridge_suite.git && \
cd rosbridge_suite && \
git fetch && git checkout humble && git pull && \
cd ../.. && \
source /opt/ros/humble/setup.bash && \
colcon build