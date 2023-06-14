#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>    
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <complex>
#include <algorithm>
#include "test_utils.h"
#include "unisocai.h"
#include <signal.h>
#include <regex>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef __LINUX__
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#else
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#endif

#include "obj_detect.h"
#include "deep_estiminate.h"

// deep extimination parameters
extern int show_flow = -1;
extern cv::Mat cameraMatrixL,distCoeffL,cameraMatrixR,distCoeffR;
extern cv::Mat Rl, Rr, Pl, Pr, Q ,R, T;                            //校正旋转矩阵R，投影矩阵P, 重投影矩阵Q
extern const int imageWidth,imageHeight;
extern cv::Size imageSize;
extern cv::Rect validROIL,validROIR;
extern cv::Mat mapLx, mapLy, mapRx, mapRy;     
extern cv::Mat frame, frame_L, frame_R;
extern cv::Mat rgbImageL, grayImageL;
extern cv::Mat rgbImageR, grayImageR;
extern cv::Mat rectifyImageL, rectifyImageR;

// obg_detection parameters
extern const float prob_threshold = 0.05f; // 0.25
extern const float nms_threshold = 0.45f;
extern const float mean_vals[3] = {0, 0, 0};                               // RGB
extern const float norm_vals[3] = {1 / (255.f), 1 / (255.f), 1 / (255.f)}; // RGB
extern const int strides[3] = {8, 16, 32};

extern const int bbox_cell = pow(INPUT_SIZE / strides[0], 2) + pow(INPUT_SIZE / strides[1], 2) + pow(INPUT_SIZE / strides[2], 2);
extern const std::vector<std::vector<int>> input_shapes = {{1, 3, INPUT_SIZE, INPUT_SIZE}};                  // NCHW
extern const std::vector<std::vector<int>> output_shapes = {{1, 4, bbox_cell}, {1, NUM_CLASSES, bbox_cell}}; // NCHW
extern const std::vector<std::string> input_node_name = {"input"};
extern const std::vector<std::string> output_node_name = {"box", "cls"};

extern unsigned int input_count = input_shapes.size();
extern unsigned int output_count = output_shapes.size();
extern void **inputbufs;
extern void **outputbufs;
extern static int keepRunning = 1;
extern ModelMgr *modelManager;
extern DataFormat dataformat;
extern NodeShape *input_node;
extern cv::Mat imgbase;
extern cv::Mat preprocess_img;
extern TestCli cli;
extern std::string file_name;
extern float img_scale; // image scaling
extern size_t label_count_;
extern std::vector<std::string> vec_labels_;
extern int img_w = 0;
extern int img_h = 0;
extern int NH = 0;
extern int NW = 0;
extern std::vector<cv::Rect> boxes;
extern std::vector<int> clsIdx_vec;   // 目标框类别索引
extern std::vector<float> scores_vec; // 目标框得分索引
extern std::vector<int> boxIdx_vec;   // 目标框的序号索引


int main(int argc, char* argv[]){

    //--立体校正-------------------------------------------------------------------
	// cv::Rodrigues(rec, R);                                   //Rodrigues变换
    cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY,
                      0, imageSize, &validROIL, &validROIR);
    cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
    cv::VideoCapture cap;

    // open camera 
    std::cout<<"\033[34m"<<"LOG: attempt to open camera\n"<<"\033[0m";
    do{cap.open("/dev/video0")}
    while(!cap.isOpened()){
        std::cout<<"\033[31"<<"open camera fail attempt open again"<<"\033[0m";
        cap.open("/dev/video0");    //【可能需要修改的程序参数4】：打开端口1对应的设备，电脑自带摄像头一般编号为0，外接摄像头编号为1，也可能是反过来的
        sleep(1);
    }
    std::cout<<"\033[34m]"<<"LOG: open camera successfully"<<"\033[0m";

	cap.set(cv::CAP_PROP_FRAME_WIDTH, imageWidth * 2); //设置捕获图像的宽度，为双目图像的宽度
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, imageHeight);  //设置捕获图像的高度
    
    if (aisdk_init(argc, argv) == -1){
        std::cout<<"\033[31m"<<"ERROR:"<<"\033[0m"<<" something errror while initial parameters\n";
        return -1;
    }

	double sf;
	int w, h;
    while (true)
    {
        cap >> frame;
        frame_L = frame(cv::Rect(0, 0, imageWidth, imageHeight));             //获取左Camera的图像
        frame_R = frame(cv::Rect(imageWidth, 0, imageWidth, imageHeight));   //获取右Camera的图像

        // turn rgb to gray 
		cv::cvtColor(frame_L, grayImageL, cv::COLOR_BGR2GRAY);
		cv::cvtColor(frame_R, grayImageR, cv::COLOR_BGR2GRAY);

        // do stereoscopic correction
		cv::remap(grayImageL, rectifyImageL, mapLx, mapLy, cv::INTER_LINEAR);
		cv::remap(grayImageR, rectifyImageR, mapRx, mapRy, cv::INTER_LINEAR);
        
        // cut images to show on the same image
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
		h = cvRound(imageSize.height * sf);

        // dispaly the images
        cv::imshow("frame_l",frame_L);
        if (cv::waitKey(1)==17)return 0; // if press excape finished the programe
        // using SGBM to estiminate depth map
        cv::Mat depth_mat = stereo_match_sgbm(0,0);

        // get img and run model detection
        get_img<float>(reinterpret_cast<float*>(inputbufs[0]), input_shapes[0], false, false);
		RunModel(modelManager, inputbufs, input_count, &dataformat,outputbufs, output_count, 1000);
		std::cout << "RunModel done" << std::endl;

        // NMS process
		select_bbox(outputbufs,img_w,img_h,img_scale);

        // draw rect on image 
		show_rect(depth_mat);
        
        if(show_flow>=0){
            cv::imshow("result", imgbase);
            
            auto k=cv::waitKey(show_flow);
            if(k == 32) { //Press space to pause
                cv::waitKey(0);
            }
        }

    }
}
