# ranging_package

## Description:
ROS2 package to perform object vehicle ranging using YOLO object detection and monocular ranging methods.

The methods used for ranging are a simple pinhole camera model which assumes a known vehicle width and a virtual horizon model which generates a "vanishing point" on the image plane for ranging - the virtual horizon method avoids setting an assumed size for all vehicles but still uses an average vehicle width to calculate the vanishing point.

**For more information on these methods and/or this package, check the Joplin Documentation under the _Ranging Package_ page**

## Requirements

* Ubuntu Focal Fossa
* ROS2 Foxy Fitzroy
* C++17 or higher

## To Use:

**_Before Use:_**
* **Make sure ALL PATHS ARE SET CORRECTLY in the launch and config files before use!**
* **These steps assume you have already created a workspace folder and a `/src` directory within it!**

**_Steps:_**
1. Navigate into the `/src` directory of your workspace and clone the repo using `git clone`
2. Navigate back into the workspace directory and source `$ source /opt/ros/foxy/setup.bash`
3. Build package `$ colcon build` or `$ colcon build --packages-select <package_name>`
4. Open a new terminal and source it `$ . install/setup.bash`
5. Run launch file `$ ros2 launch <package_name> <launch_file_name>` in this case it is `$ ros2 launch ranging_pacakge object_position_estimation_launch.py`