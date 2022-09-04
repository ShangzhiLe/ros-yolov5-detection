#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

// a struct including class_id, confidence and bounding box
struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

ov::CompiledModel openvino_init(){
    ov::Core core;
    //Read a model
    std::shared_ptr<ov::Model> model = core.read_model("/home/le/detect_yolov5_ws/src/yolov5/yolov5s/yolov5m.onnx");

    //Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);

    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});

    ppp.input().model().set_layout("NCHW");
 
    ppp.output().tensor().set_element_type(ov::element::f32);
 
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    return compiled_model;
}
void openvino_detect(const sensor_msgs::ImageConstPtr& msg){
    //Read input image
    //cv::Mat img = cv::imread("/home/le/detect_yolov5_ws/src/yolov5/yolov5s/zidane.jpg");
    cv::Mat frame = cv_bridge::toCvCopy(msg,"bgr8")->image;
    // resize image to 640*640 which is the input size of yolov5 model
    cv::Mat newFrame = cv::Mat::zeros(640, 640, CV_32FC3);
    cv::resize(frame, newFrame,newFrame.size());

    ov::CompiledModel compiled_model = openvino_init();
    // Create tensor from image
    float *input_data = (float *) newFrame.data;
    ov::Tensor input_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);


    // Create a infer request 
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();//start inferring


    // Get output results 
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape(); //get the output shape, should be 1*25200*85 here
    //std::cout<<"output_shape 1:"<<output_shape[1]<<std::endl;
    float *detections = output_tensor.data<float>();

    //load class names
    std::ifstream readFile("/home/le/detect_yolov5_ws/src/yolov5/yolov5s/coco.names");
    std::string label;
    std::vector<std::string> labels;
    while(getline(readFile,label)){
        labels.push_back(label);
    }

    // Postprocessing including NMS  
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    // exam for every pixel
    for (int i = 0; i < output_shape[1]; i++){
        float *detection = &detections[i * output_shape[2]]; 

        float confidence = detection[4];//get the confidence
        if (confidence >= CONFIDENCE_THRESHOLD){
            float *classes_scores = &detection[5]; //score
            cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);//the first 5 numbers are x,y,w,h,confidence
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD){

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                float xmin = x - (w / 2);
                float ymin = y - (h / 2);

                boxes.push_back(cv::Rect(xmin, ymin, w, h));
            }
        }
    }
    std::vector<int> nms_result;

    //NMSBoxes method
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<Detection> output;

    //record all the vaild results in Struct Detection
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }


    // Print results and save Figure with detections
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;
        // std::cout << "Bounding box" << i + 1 << ": Class: " << labels[classId] << " "
        //      << "Confidence: " << confidence << " Scaled coords: [ "
        //      << "cx: " << (float)(box.x + (box.width / 2)) / newFrame.cols << ", "
        //      << "cy: " << (float)(box.y + (box.height / 2)) / newFrame.rows << ", "
        //      << "w: " << (float)box.width / newFrame.cols << ", "
        //      << "h: " << (float)box.height / newFrame.rows << " ]" << std::endl;
        float xmax = box.x + box.width;
        float ymax = box.y + box.height;
        cv::rectangle(newFrame, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 0, 255), 3);
        //cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(newFrame, labels[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0));
        
    }
    ROS_INFO("open successfully");
    cv::imshow("Output", newFrame);
    cv::waitKey(1);

    //cv::imwrite("/home/le/detect_yolov5_ws/src/yolov5/yolov5s/detection.png", newFrame);

}

int main(int argc, char** argv){
    ros::init(argc,argv,"listener");
    ros::NodeHandle nh;
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("image", 1, openvino_detect);
    ros::spin();
    if (cv::waitKey(1)  == 27)
        cv::destroyWindow("Output");
    //openvino_detect();
    return 0;
}