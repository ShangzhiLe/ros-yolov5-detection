#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_opencv");
    // 声明节点
    ros::NodeHandle nh;
    // image_transport image publish and subscribe
    image_transport::ImageTransport it(nh);
    
    image_transport::Publisher pub = it.advertise("image", 1);
    
    cv::VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cerr << "Read video Failed !" << std::endl;
        return 0;
    }

    cv::Mat image;
    while(ros::ok() && cap.isOpened()){
        ROS_INFO("system is fine");
        cap.read(image);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        msg->header.stamp = ros::Time::now();
        ros::Rate loop_rate(5);
        pub.publish(msg);
        ros::spinOnce();
        
        loop_rate.sleep();
    }
    cap.release();
}