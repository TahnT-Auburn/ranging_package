from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    config_path = "/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/ranging_package/config/object_position_estimation.yaml"
    
    return LaunchDescription([
        Node(
            package="ranging_package",
            executable="yolo_object_position_estimation",
            name="object_position_estimation",
            output="screen",
            emulate_tty=True,
            parameters=[config_path]
        ),
    ])