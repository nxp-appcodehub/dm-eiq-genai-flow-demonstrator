#!/bin/bash

# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

# Package installation and setup script for ROS 2 with GenAI support
./install.sh
pip install build
rm -rf build dist **/*.egg-info
python3 -m build --wheel
pip install dist/eiq_genai_flow-*.whl

# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash

# Create workspace
root_dir=$(pwd)
mkdir -p ros2_ws/src
cd ros2_ws/src

# Create the package with dependencies
ros2 pkg create --build-type ament_python imx_genai \
  --dependencies rclpy std_msgs std_srvs

# Copy ros node
cd ${root_dir}
cp ros_node.py ros2_ws/src/imx_genai/imx_genai/flow.py

# Modify setup with all the information
cat > ros2_ws/src/imx_genai/setup.py << 'EOF'
from setuptools import setup

package_name = 'imx_genai'

setup(
    name=package_name,
    version='3.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NXP AI SW Team',
    maintainer_email='voice@nxp.com',
    description='ROS 2 wrapper for eIQ GenAI Flow',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'eiq_genai_flow = imx_genai.flow:main',
        ],
    },
)
EOF

# Build package
cd ros2_ws
colcon build --packages-select imx_genai
cd ${root_dir}

# Source the workspace
source ros2_ws/install/setup.bash

echo "ROS 2 eIQ GenAI Flow node setup complete!"
echo "You can now run the node, ex:" 
echo "ros2 run imx_genai eiq_genai_flow"
