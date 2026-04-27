cd ws_aic/src/aic

# Ensure
pixi install
source /opt/ros/kilted/setup.bash

# Install aic_description and other tesseract package dependencies
# Tesseract Installation

colcon build --packages-select aic_assets aic_description --packages-ignore aic_gazebo aic_engine_interfaces --symlink-install --executor sequential
source install/setup.bash
pixi run python -m pip install tesseract-robotics==0.5.1

# ONLY for indexing in vscode when new source code is created

pixi reinstall ros-kilted-aic-model

# Test policy without gazebo gui
/entrypoint.sh ground_truth:=false start_aic_engine:=true gazebo_gui:=false
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.TestMove