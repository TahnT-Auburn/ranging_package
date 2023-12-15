//STD
#include <fstream>
#include <sstream>
#include <iostream>
#include <tuple>
#include <numeric>
#include <complex>

//OpenCV
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

//ROS2
#include "yolo_object_position_estimation.h"

using namespace cv;
using namespace dnn;
using namespace std;

void YoloObjectPositionEstimation::parseParameters()
{
    //Yolo parameters
    this->declare_parameter("conf_threshold");
    this->declare_parameter("nms_threshold");
    this->declare_parameter("inp_width");
    this->declare_parameter("inp_height");
    this->declare_parameter("device");
    this->declare_parameter("input_type");
    this->declare_parameter("input_path");
    this->declare_parameter("write_output");
    this->declare_parameter("output_file");
    this->declare_parameter("classes_file");
    this->declare_parameter("model_configuration");
    this->declare_parameter("model_weights");
    //Camera intrinsics
    this->declare_parameter("camera_matrix_vector");
    this->declare_parameter("dist_coeffs");
    //Camera params
    this->declare_parameter("camera_height");

    //Get Yolo params
    this->get_parameter("conf_threshold", confThreshold);   
    this->get_parameter("nms_threshold", nmsThreshold);
    this->get_parameter("inp_width", inpWidth);
    this->get_parameter("inp_height", inpHeight);
    this->get_parameter("device", device_);
    this->get_parameter("input_type", input_type_);
    this->get_parameter("input_path", input_path_);
    this->get_parameter("write_output", write_output_);
    this->get_parameter("output_file", output_file_);
    this->get_parameter("classes_file", classesFile);
    this->get_parameter("model_configuration", modelConfiguration);
    this->get_parameter("model_weights", modelWeights);
    //Get camera intrinsics
    this->get_parameter("camera_matrix_vector", camera_matrix_vector_);
    this->get_parameter("dist_coeffs", dist_coeffs_);   
    //Get camera params
    this->get_parameter("camera_height", camera_height_);

    //Check
    RCLCPP_INFO(this->get_logger(), "Confidence Threshold: %f", confThreshold);
    RCLCPP_INFO(this->get_logger(), "Non-Maximum Suppression Threshold: %f", nmsThreshold);
    RCLCPP_INFO(this->get_logger(), "Input Width: %d", inpWidth);
    RCLCPP_INFO(this->get_logger(), "Input height: %d", inpHeight);
    RCLCPP_INFO(this->get_logger(), "Device: %s", device_.c_str());
}

void YoloObjectPositionEstimation::run()
{   

    //Load Darknet
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device_ == "cpu")
    {
        RCLCPP_INFO(this->get_logger(), "Using %s device", device_.c_str());
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device_ == "gpu")
    {   
        RCLCPP_INFO(this->get_logger(), "Using %s device", device_.c_str());
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }

    //Read image or video data
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    cv::Mat frame, blob;

    try
    {
        //outputFile = "yolo_out_cpp.jpg";
        if (input_type_ == "image")
        {
            //Open image file
            ifstream ifile(input_path_);
            if (!ifile)
            {
                RCLCPP_ERROR(this->get_logger(),"No image found");
            }
            cap.open(input_path_);
            outputFile = output_file_;
            RCLCPP_INFO(this->get_logger(), "image opened successfully\n");
        }
        else if (input_type_ == "video")
        {
            //Open video file
            ifstream ifile(input_path_);
            if (!ifile)
            {
                RCLCPP_ERROR(this->get_logger(),"No video found");
            }
            cap.open(input_path_);
            outputFile = output_file_;
        }
    }
    catch(...)
    {
        RCLCPP_ERROR(this->get_logger(), "Could not open input file");
        rclcpp::shutdown();
    }
    
    // Get the video writer initialized to save the output video
    if (input_type_ == "video")
    {   
        double fps = cap.get(CAP_PROP_FPS);
        RCLCPP_INFO(this->get_logger(), "fps: %f", fps);
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), fps, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }

    //Process frames
    while (true)
    {
        // get frame from the video
        cap >> frame;
        //frame = imread(input_path_, IMREAD_COLOR);

        // Stop the program if reached end of video
        if (frame.empty()) {
            RCLCPP_INFO(this->get_logger(), "Processing completed successfully");
            if (write_output_)
            {
                RCLCPP_INFO(this->get_logger(), "Output writen to: %s", output_file_.c_str());
            }
            waitKey(3000);
            break;
        }

        //Undistort frame
        cv::Mat undist_frame = undistortImage(frame);

        // Create a 4D blob from a frames
        blobFromImage(undist_frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Run post process funtion
        /*
        Non-maximun suppression to isolate best bounding box
        Range estimation
        Bearing estimation
        Lateral offset estimation
        Drawing on image using OpenCV
        */
        postProcess(undist_frame, outs);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Processing Time : %.2f ms", t);
        putText(undist_frame, label, Point(0, frame.rows-15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.75);
        
        //Draw reference axis at optical center
        cv::arrowedLine(undist_frame, cv::Point(frame.cols/2, frame.rows), cv::Point(frame.cols/2, frame.rows-75), cv::Scalar(255,255,255), 2.25);
        cv::arrowedLine(undist_frame, cv::Point(frame.cols/2, frame.rows-1.5), cv::Point(frame.cols/2 + 75, frame.rows-1.5), cv::Scalar(255,255,255), 2.25);

        // Write the frame with the detection boxes
        if (write_output_)
        {
            cv::Mat detectedFrame;
            undist_frame.convertTo(detectedFrame, CV_8U);
            if (input_type_ == "image")
            {
                imwrite(output_file_, detectedFrame);
            }
            else
            {
                video.write(detectedFrame);
            }

        }

        //Display
        imshow("Detection", undist_frame);
        char key = (char) waitKey(1);
        if (key == 27)
        {   
            break;
            destroyAllWindows();
            rclcpp::shutdown();
        }
    }
    cap.release();
    if (input_type_ == "video") video.release();   
}

cv::Mat YoloObjectPositionEstimation::undistortImage(const Mat& frame)
{   
    //Set camera matrix and distortion coefficients
    
    cv::Matx33d camera_matrix_(camera_matrix_vector_[0], camera_matrix_vector_[1], camera_matrix_vector_[2], 
                            camera_matrix_vector_[3], camera_matrix_vector_[4], camera_matrix_vector_[5], 
                            camera_matrix_vector_[6], camera_matrix_vector_[7], camera_matrix_vector_[8]);


    //Set image size
    int width_ = frame.cols;
    int height_ = frame.rows;
    cv::Size frame_size_(width_, height_);

    //Precompute lens correction interpolation
    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Matx33f::eye(), camera_matrix_, frame_size_, CV_32FC1, mapX, mapY);

    //Undistort image
    cv::Mat undist_frame;
    cv::remap(frame, undist_frame, mapX, mapY, cv::INTER_LINEAR);

    return undist_frame;
}

void YoloObjectPositionEstimation::postProcess(const Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    //Load names of classes
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    //Initialize bounding box vectors - used to track vehicle objects only (First detection filtering step)
    vector<int> indices;
    vector<int> idx_vector = {};
    vector<int> obj_class_id_vector = {};
    vector<double> conf_vector = {};
    vector<int> left_vector = {};
    vector<int> top_vector = {};
    vector<int> right_vector = {};
    vector<int> bottom_vector = {};
    vector<int> width_vector = {};
    vector<int> height_vector = {};

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    //Filter for vehicle detections only
    int init_count = 0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        int obj_class_id = classIds[idx];
        double conf = confidences[idx];
        string obj_class = classes[classIds[idx]];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int right = box.x + box.width;
        int bottom = box.y + box.height;
        int box_width = box.width;
        int box_height = box.height;

        
        //Filter out non-vehicle class objects
        if (obj_class == "car" || obj_class == "truck" || obj_class == "bus")
        {   
            
            init_count++;
            //Track vehicle detection info and bounding boxes for horizon estimation
            idx_vector.push_back(idx);
            obj_class_id_vector.push_back(obj_class_id);
            conf_vector.push_back(conf);
            left_vector.push_back(left);
            top_vector.push_back(top);
            right_vector.push_back(right);
            bottom_vector.push_back(bottom);
            width_vector.push_back(box_width);
            height_vector.push_back(box_height);
        }
        else
        {
            drawPred(false, false, idx, classIds[idx], confidences[idx], left, top,
                right, bottom, box_width, frame);
        }

    }

    //Get average
    double avg_bottom = getAverage(bottom_vector);
    double avg_box_width = getAverage(width_vector);

    //Call horizon estimation
    double virtual_horizon = horizonEstimation(avg_bottom, avg_box_width, frame);

    //Initialize variables set for 
    double range_ss_x;
    double range_ss_y;
    double range_vh_x;
    double range_vh_y;
    double range_ss_norm;
    double range_vh_norm;
    double bearing_ss;
    double bearing_vh;
    std::complex<double> comp_ss;
    std::complex<double> comp_vh;
    int val_count = 0;
    int inval_count = 0;
    vector<int> bottom_vector_val = {};
    vector<int> width_vector_val = {};
    vector<int> val_count_vector = {};

    //Hard set loop for virtual horizon to reach steady-state
    int k = 0;
    while (k < 5) 
    {   
        RCLCPP_INFO(this->get_logger(), "virtual horizon: %f", virtual_horizon);

        //Reset valid vector
        bottom_vector_val = {};
        width_vector_val = {};


        //Iterate through vehicle detections
        for (size_t i = 0; i < idx_vector.size(); ++i)
        {   

            //Static size ranging
            range_ss_x = rangeEstimation(width_vector[i]);
            auto [range_ss_y, bearing_ss] = bearingEstimation(right_vector[i], width_vector[i], range_ss_x, frame);
            std::complex<double> comp_ss(range_ss_x, range_ss_y);
            range_ss_norm = sqrt(norm(comp_ss));
            if (k == 4)
            {   
                RCLCPP_INFO(this->get_logger(), "-----Static Ranging-----\n");
                RCLCPP_INFO(this->get_logger(), "Tag: %d", idx_vector[i]);
                RCLCPP_INFO(this->get_logger(), "ss range x: %f", range_ss_x);
                RCLCPP_INFO(this->get_logger(), "ss range y: %f", range_ss_y);
                RCLCPP_INFO(this->get_logger(), "ss range norm: %f", range_ss_norm);
                RCLCPP_INFO(this->get_logger(), "ss bearing: %f\n", bearing_ss);
            }

            //Virtual horizon ranging
            range_vh_x = rangeFromHorizon(bottom_vector[i], virtual_horizon);
            auto [range_vh_y, bearing_vh] = bearingEstimation(right_vector[i], width_vector[i], range_vh_x, frame);
            std::complex<double> comp_vh(range_vh_x, range_vh_y);
            range_vh_norm = sqrt(norm(comp_vh));
            if (k == 4)
            {
                RCLCPP_INFO(this->get_logger(), "-----Horizon Ranging-----\n");
                RCLCPP_INFO(this->get_logger(), "Tag: %d", idx_vector[i]);
                RCLCPP_INFO(this->get_logger(), "vh range x: %f", range_vh_x);
                RCLCPP_INFO(this->get_logger(), "vh range y: %f", range_vh_y);
                RCLCPP_INFO(this->get_logger(), "vh range norm: %f", range_vh_norm);
                RCLCPP_INFO(this->get_logger(), "vh bearing: %f\n", bearing_vh);
            }

            //Detection filter
            bool valid_detection = detectionFilter(bottom_vector[i], width_vector[i], virtual_horizon);

            if (valid_detection)
            {
                val_count++;

                //Populate valid detection bounding box dimensions to recalculate virtual horizon
                bottom_vector_val.push_back(bottom_vector[i]);
                width_vector_val.push_back(width_vector[i]);

                drawPred(true, true, idx_vector[i], obj_class_id_vector[i], conf_vector[i], left_vector[i], top_vector[i],
                    right_vector[i], bottom_vector[i], width_vector[i], frame);
            }
            else
            {
                inval_count++;
                drawPred(true, false, idx_vector[i], obj_class_id_vector[i], conf_vector[i], left_vector[i], top_vector[i],
                    right_vector[i], bottom_vector[i], width_vector[i], frame);
            }

        }   

        //Update averages of valid vectors
        avg_bottom = getAverage(bottom_vector_val);
        avg_box_width = getAverage(width_vector_val);

        //Update virtual horizon
        virtual_horizon = horizonEstimation(avg_bottom, avg_box_width, frame);

        k++;

    }
}

void YoloObjectPositionEstimation::drawPred(bool vehicle_detection, bool valid_detection, int idx, int classId, float conf, int left, int top, int right, int bottom, int box_width, const Mat& frame)
{   
    //Load names of classes
    vector<string> classes;    
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    //Draw green box for valid vehicle detections
    if (vehicle_detection && valid_detection)
    {   
        //Draw a rectangle displaying the bounding box
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 65), 1.5); //Scalar(255, 178, 50)
        
        //Draw line from optical center to center of bounding box
        cv::Point bounding_box_center = cv::Point(right - (box_width / 2), bottom);
        cv::line(frame, cv::Point(frame.cols/2, frame.rows), bounding_box_center, cv::Scalar(0, 255, 65), 1.5);

        //Get the label for the class name and its confidence
        string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = "Tag:" + to_string(idx) + " " + classes[classId] + ":" + label;
        }
        
        //Display the class name and confidence label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top1 = max(top, labelSize.height);
        rectangle(frame, Point(left, top1 - round(1*labelSize.height)), Point(left + round(1*labelSize.width), top1 + baseLine), Scalar(0, 255, 65), FILLED);
        putText(frame, label, Point(left, top1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0),1);
    }
    //Draw red box for non vehicle detections (must also be declared invalid)
    if (!vehicle_detection && !valid_detection)
    {   
        //Draw a rectangle displaying the bounding box
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(49, 49, 255), 1.5); //Scalar(255, 178, 50)
        
        //Get the label for the class name and its confidence
        string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = "Tag:" + to_string(idx) + " " + classes[classId] + ":" + label;
        }
        
        //Display the class name and confidence label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top1 = max(top, labelSize.height);
        rectangle(frame, Point(left, top1 - round(1*labelSize.height)), Point(left + round(1*labelSize.width), top1 + baseLine), Scalar(49, 49, 255), FILLED);
        putText(frame, label, Point(left, top1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0),1);
    }
    //Draw yellow box for invalid vehicle detections
    if (vehicle_detection && !valid_detection)
    {   
        //Draw a rectangle displaying the bounding box
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(34, 231, 224), 1.5); //Scalar(255, 178, 50)

        //Draw line from optical center to center of bounding box
        cv::Point bounding_box_center = cv::Point(right - (box_width / 2), bottom);
        cv::line(frame, cv::Point(frame.cols/2, frame.rows), bounding_box_center, cv::Scalar(34, 231, 224), 1.5);

        //Get the label for the class name and its confidence
        string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = "Tag:" + to_string(idx) + " " + classes[classId] + ":" + label;
        }
        
        //Display the class name and confidence label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top1 = max(top, labelSize.height);
        rectangle(frame, Point(left, top1 - round(1*labelSize.height)), Point(left + round(1*labelSize.width), top1 + baseLine), Scalar(34, 231, 224), FILLED);
        putText(frame, label, Point(left, top1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0),1);
    }
}

vector<String> YoloObjectPositionEstimation::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

double YoloObjectPositionEstimation::rangeEstimation(int box_width)
{
    //Camera focal length (in pixels)
    double focal_length = camera_matrix_vector_[0];

    //object size (starting with hard code) (in meters)
    double obj_width = 1.8034;

    //Calculate ranges from width 
    double range_x = focal_length * (obj_width / box_width);

    return range_x;
}

tuple<double, double> YoloObjectPositionEstimation::bearingEstimation(int right, int box_width, double range, const Mat& frame)
{   
    //Focal length of camera (in pixels)
    double focal_length = camera_matrix_vector_[0];

    //center of bounding box in the x axis
    int box_center_x = right - (box_width / 2);

    //Lateral error from image center to bounding box center
    int image_center_x = frame.cols / 2;   //Image center in x axis
    int lat_offset_img = box_center_x - image_center_x;

    //Re-project to lateral offset in world units
    double range_y = lat_offset_img * range / focal_length;

    //Bearing angle
    double pi = 3.141592653589793238463;
    double bearing_angle = (180 / pi) * atan(lat_offset_img/focal_length);

    return {range_y, bearing_angle};
}

double YoloObjectPositionEstimation::horizonEstimation(int avg_bottom, int avg_box_width, const Mat& frame)
{   
    //Set average vehicle width
    double avg_veh_width = 1.82;

    //Calculate virtual horizon
    double virtual_horizon = avg_bottom - (camera_height_ * (avg_box_width/avg_veh_width));
    
    //Eliminate invalid solutions
    if (virtual_horizon < 0 || virtual_horizon > frame.rows)
    {
        virtual_horizon = 0.0;
    }
    
    return virtual_horizon;
}

double YoloObjectPositionEstimation::rangeFromHorizon(int bottom, double virtual_horizon)
{
    //Focal length of camera (in pixels)
    double focal_length = camera_matrix_vector_[0];

    //Range from horizon
    double range_horizon = (focal_length * camera_height_) / (bottom - virtual_horizon);

    return range_horizon;
}

bool YoloObjectPositionEstimation::detectionFilter(int bottom, int box_width, double virtual_horizon)
{
    //Define lower limit
    double min_veh_width = 1.4;//1.4
    double lower_limit = ((bottom - virtual_horizon) / camera_height_) * min_veh_width;

    //Define upper limit
    double max_veh_width = 2.6;//2.6;
    double upper_limit = ((bottom - virtual_horizon) / camera_height_) * max_veh_width;

    //Define valid detection
    bool valid_detection;
    if (box_width >= lower_limit && box_width <= upper_limit)
    {
        valid_detection = true;
    }
    else
    {
        valid_detection = false;
    }

    return valid_detection;
}

double YoloObjectPositionEstimation::getAverage(vector<int> vect)
{
    if (vect.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "Input vector is empty");
    }
    return std::accumulate(vect.begin(), vect.end(), 0.0) / vect.size();
}  