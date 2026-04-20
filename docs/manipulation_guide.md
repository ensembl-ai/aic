cd ws_aic/src/aic

# Install aic_description and other tesseract package dependencies
# Tesseract Installation

source /opt/ros/kilted/setup.bash
pixi shell
colcon build --packages-select aic_assets aic_description --executor sequential
source install/setup.bash
python -m pip install tesseract-robotics==0.5.1

# For indexing in vscode

pixi reinstall ros-kilted-aic-model