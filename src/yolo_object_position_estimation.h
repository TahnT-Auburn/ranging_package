//STD
#include <fstream>
#include <sstream>
#include <iostream>

//OpenCV
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//ROS2
#include <rclcpp/rclcpp.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class YoloObjectPositionEstimation : public rclcpp::Node
{
public:
    YoloObjectPositionEstimation()
    : Node("yolo_object_position_estimation")
    {   
        parseParameters();
        run();
        //undistortImage();
    }

private:

    //Initialize ROS2 parameters
    float confThreshold;    //YOLO
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    string device_;
    string input_type_;
    string input_path_;
    bool write_output_;
    string output_file_;
    string classesFile;
    String modelConfiguration;
    String modelWeights;
    std::vector<double> camera_matrix_vector_;  //Cam intrinsics
    std::vector<double> dist_coeffs_;
    double camera_height_;  //Cam params

    //Class methods
    void parseParameters();
    void run();
    cv::Mat undistortImage(const Mat& frame);
    void postProcess(const Mat& frame, const vector<Mat>& outs);
    void drawPred(bool vehicle_detection, bool valid_detection, int idx, int classId, float conf, int left, int top, int right, int bottom, int box_width, const Mat& frame);
    vector<String> getOutputsNames(const Net& net);
    double rangeEstimation(int box_width);
    tuple<double,double> bearingEstimation(int right, int box_width, double range, const Mat& frame);
    double horizonEstimation(int avg_bottom, int avg_box_width, const Mat& frame);
    double rangeFromHorizon(int bottom, double virtual_horizon);
    bool detectionFilter(int bottom, int box_width, double virtual_horizon);
    double getAverage(vector<int> vect);
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloObjectPositionEstimation>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}