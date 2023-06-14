// File Name:main.cpp
// Date:     2019-07-09
// Copyright (C) 2019 UNISOC Technologies Co.,Ltd. All Rights Reserved
#include <deep_estiminate.h>
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cassert>
#include <cstring>
#include <cmath>
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
//#include "logger.h"

#ifdef __LINUX__
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#else
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#endif
// #define MODE "" 
#define LOG_TAG "UNISOC_AISDK"
#define FLOAT32 "float32"
#define Q8A "Q8A"

#define INPUT_SIZE 640
#define NUM_CLASSES 1 
#define OUTPUT_SIZE_W 800
#define OUTPUT_SIZE_H 600
// #define MIN(x,y) x<y?x:y
// #define MMAX(x,y) x>y?x:y
//video camera path or number path
// const std::string video_number="/dev/video0";//"test.mp4"
const std::string video_number="/home/unisoc/Desktop/yolov5/model_deploy/image/VID20230525234333.mp4";//"test.mp4"
//const std::string image_folder="/home/unisoc/test/images";//absolute path is recommended
const std::string image_folder="/home/unisoc/Desktop/yolov5/model_deploy/image";
//read data mode(viode or image folder)
const int read_mode=0;// 0 video; other:local image folder
//show data flow
const int show_flow=-1;// <0 not show, 0 fram or >0 stream interval time(mm);

const float prob_threshold = 0.05f; // 0.25
const float nms_threshold = 0.45f;
const float mean_vals[3] = {0, 0, 0};                               // RGB
const float norm_vals[3] = {1 / (255.f), 1 / (255.f), 1 / (255.f)}; // RGB
const int strides[3] = {8, 16, 32};

const int bbox_cell = pow(INPUT_SIZE / strides[0], 2) + pow(INPUT_SIZE / strides[1], 2) + pow(INPUT_SIZE / strides[2], 2);
const std::vector<std::vector<int>> input_shapes = {{1, 3, INPUT_SIZE, INPUT_SIZE}};                  // NCHW
const std::vector<std::vector<int>> output_shapes = {{1, 4, bbox_cell}, {1, NUM_CLASSES, bbox_cell}}; // NCHW
const std::vector<std::string> input_node_name = {"input"};
const std::vector<std::string> output_node_name = {"box", "cls"};

unsigned int input_count = input_shapes.size();
unsigned int output_count = output_shapes.size();
void **inputbufs;
void **outputbufs;
static int keepRunning = 1;
ModelMgr *modelManager;
DataFormat dataformat;
NodeShape *input_node;
cv::Mat imgbase;
cv::Mat preprocess_img;
TestCli cli;
std::string file_name;
float img_scale; // image scaling
size_t label_count_;
std::vector<std::string> vec_labels_;
int img_w = 0;
int img_h = 0;
int NH = 0;
int NW = 0;
std::vector<cv::Rect> boxes;
std::vector<int> clsIdx_vec;   // 目标框类别索引
std::vector<float> scores_vec; // 目标框得分索引
std::vector<int> boxIdx_vec;   // 目标框的序号索引

template <class T>
void hwc_to_chw(T *dst, T *src, int height, int width, int channel, const float *mean_vals, const float *norm_vals)
{
    for (int cc = 0; cc < channel; cc++)
    {
        for (int hw = 0; hw < (width * height); hw++)
        {
            dst[cc * height * width + hw] = (src[hw * channel + cc] - mean_vals[cc]) * norm_vals[cc];
        }
    }
}
template <class T>
void free_2d_arr(T **arr2d, int size)
{
    if (arr2d)
    {
        for (int i = 0; i < size; ++i)
        {
            if (arr2d[i])
            {
                std::free(arr2d[i]);
            }
        }
        std::free(arr2d);
    }
}
template <class T>
void get_img(T *imgData_buffer,             // [out] input buffer
             const std::vector<int> &shape, // [in] input shape
             bool is_nhwc = false,          // [in] true: NHWC; false: NCHW
             bool is_bgr_mode = false)
{ // [in] true: BGR; false: RGB
    if (imgbase.empty())
    {
        std::cout << "*****OpenCV::imread error! - read " << std::endl;
        return;
    }
    int wanted_channels = shape[1];
    int wanted_height = shape[2];
    int wanted_width = shape[3];
    auto cv_fc3 = CV_8UC3;
    auto cv_fc1 = CV_8UC1;
    if (typeid(T) == typeid(float))
    {
        cv_fc3 = CV_32FC3;
        cv_fc1 = CV_32FC1;
    }

    auto cv_fc = cv_fc3;
    if (wanted_channels == 1)
    {
        cv_fc = cv_fc1;
    }
    cv::Mat img = imgbase;

    cv::Mat sample;
    if (img.channels() == 4)
    {
        auto colorConv = cv::COLOR_BGRA2RGB;
        if (is_bgr_mode)
        {
            colorConv = cv::COLOR_BGRA2BGR;
        }
        cv::cvtColor(img, sample, colorConv);
    }
    else if (img.channels() == 1)
    {
        auto colorConv = cv::COLOR_GRAY2RGB;
        if (is_bgr_mode)
        {
            colorConv = cv::COLOR_GRAY2BGR;
        }
        cv::cvtColor(img, sample, colorConv);
    }
    else if (img.channels() == 3)
    {
        if (is_bgr_mode)
        {
            sample = img;
        }
        else
        {
            cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
        }
    }
    else {
        std::cout << "wanted_channels: " <<wanted_channels
                  << ", Unknown channels: " << img.channels() << std::endl;
    }
    cv::Mat img_tmp;
    int nw=sample.cols;
    int nh=sample.rows;
    if (nw > nh){
        img_scale = (float)INPUT_SIZE / nw;
        nw = INPUT_SIZE;
        nh = nh * img_scale;
    }
    else{
        img_scale = (float)INPUT_SIZE / nh;
        nh = INPUT_SIZE;
        nw = nw * img_scale;
    }
    cv::resize(sample, img_tmp, cv::Size(nw, nh)); 
    int nh_size=(INPUT_SIZE - nh);
    int nw_size=(INPUT_SIZE - nw);
    NH=nh_size/2;
    NW=nw_size/2;
    cv::copyMakeBorder(img_tmp, sample, nh_size/2 , nh_size-nh_size/2 , nw_size/2 , nw_size-nw_size/2 ,
                         cv::BORDER_CONSTANT, cv::Scalar(114.f,114.f,114.f)); //1
    //cv::copyMakeBorder(img_tmp, sample, 0, INPUT_SIZE - nh, 0, INPUT_SIZE - nw,cv::BORDER_CONSTANT, 114.f);
    printf("%d %d\n",sample.cols,sample.rows);
    assert(sample.cols==INPUT_SIZE && sample.rows==INPUT_SIZE);

    if (typeid(T) == typeid(float)) {
        sample.convertTo(preprocess_img, cv_fc);
    } 
    else {
        preprocess_img = sample;
    }
    // conver to nchw
    hwc_to_chw<T>(imgData_buffer, reinterpret_cast<T*>(preprocess_img.data),
                  wanted_height, wanted_width, wanted_channels,mean_vals,norm_vals);
}

int ReadLabelsFile(const std::string& file_name,
                   std::vector<std::string>* result,
                   size_t* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        std::cout << "read label file: " << file_name.c_str()
                << "failed" << std::endl;
        return -1;
    }
    result->clear();
    std::string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return 0;
}


void sig_handler( int sig )
{
    if ( sig == SIGINT){
    keepRunning = 0;
    printf("sig_handler!!");
    }
}

static void get_bbox(float **outputs,float prob_threshold)
{
    boxes.clear();
    clsIdx_vec.clear();//目标框类别索引
    scores_vec.clear();//目标框得分索引
    
    float *boxes_ptr=outputs[0];
    float *clses=outputs[1];
    for (int i = 0; i < bbox_cell; i++,clses+=NUM_CLASSES,boxes_ptr+=4){
        float max_score=clses[0];
        float max_score_id=0;
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++){
            if(max_score<clses[class_idx]){
                max_score=clses[class_idx];
                max_score_id=class_idx;
            }
        } 
            if(max_score<prob_threshold){
                continue;
        }
        int x1=int((boxes_ptr[0]-NW)/img_scale);
        int y1=int((boxes_ptr[1]-NH)/img_scale);
        int x2=int((boxes_ptr[2]-NW)/img_scale);
        int y2=int((boxes_ptr[3]-NH)/img_scale);    //2
        x1=std::max(0,std::min(x1,img_w));
        y1=std::max(0,std::min(y1,img_h));
        x2=std::max(0,std::min(x2,img_w));
        y2=std::max(0,std::min(y2,img_h));
        boxes.push_back(cv::Rect(x1,y1,x2,y2));
        clsIdx_vec.push_back(max_score_id);
        scores_vec.push_back(max_score);
    } 
}

static void select_bbox(void **outputbuf,int nw,int nh,float img_scale){
    get_bbox(
        reinterpret_cast<float**>(outputbuf),
        prob_threshold
    );
    boxIdx_vec.clear(); //目标框的序号索引
    cv::dnn::NMSBoxes(boxes, scores_vec, 0.0f, nms_threshold, boxIdx_vec);
    printf("box size : %d %d %d\n",boxes.size(), clsIdx_vec.size(),scores_vec.size());
    
}
    
static void show_rect(cv::Mat depth_map){
    printf("boxIdx_vec size %d\n",boxIdx_vec.size());
    static char out_path[50];

    for (size_s i = 0; i < boxIdx_vec.size(); i++){
        int idx=boxIdx_vec[i];
        cv::Rect &rect= boxes[idx];
        int label=clsIdx_vec[idx];
        float prob=scores_vec[idx];
        fprintf(stderr, "%d = %.5f at %d %d %d %d -- %s \n", label, prob,
                rect.x, rect.y, rect.width, rect.height,rec_labels_[label].c_str());

        cv::rectangle(imgbase, cv::Rect(rect.x, rect.y, rect.width-rect.x, rect.height-rect.y), cv::Scalar(255, 0, 0));

        char text[256];
        int depth = depth_map.at<int>(cv::Point(x,y))
        sprintf(text, "%s %.1f%% %d", vec_labels_[label].c_str(), prob * 100,depth);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = rect.x;
        int y = rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > imgbase.cols)
            x = imgbase.cols - label_size.width;
        cv::rectangle(imgbase, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);
        cv::putText(imgbase, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        if (!depth_map.empty()){
            extern 
        }
    }
}

int aisdk_init(int argc, char* argv[]){
	// get AISDK version
	unsigned int sdkverlen = GetSDKVersionLength();
	char* sdkver = reinterpret_cast<char*>(malloc(sdkverlen));
	std::cout << LOG_TAG << " huangtao && xiaoyao "<< std::endl;
	if (sdkver) {
		GetSDKVersion(sdkver);
		std::cout << LOG_TAG << " Version : " << sdkver << std::endl;
		std::free(sdkver);
	} 
    else {
		std::cout << LOG_TAG << " Get SDK Version fail" << std::endl;
		return -1;
	}

	// parse cmdline
	cli.Parse(argc, argv);
	// 1. create model manager
	std::cout << LOG_TAG << " create model manager  " << std::endl;
	struct timeval init;
	gettimeofday(&init, nullptr);
    modelManager = CreateModelManager();
	if (!modelManager) {
		std::cout << LOG_TAG << " ModelManager is nullptr " << std::endl;
		return -1;
	}
	struct timeval initend;
	gettimeofday(&initend, nullptr);
	double elapsedinit = (initend.tv_sec - init.tv_sec) * 1000.0 +
						(initend.tv_usec - init.tv_usec) / 1000.0;
	std::cout << LOG_TAG << " Elapsed_Time for InitEnv : " << elapsedinit
				<< std::endl;
	// 2. load model
	struct timeval loadmodel;
	gettimeofday(&loadmodel, nullptr);
	std::cout << LOG_TAG << " load Model " << cli.modelfile().c_str()
				<< std::endl;
	int ret = LoadModel(modelManager, cli.modelfile().c_str(), HIGH_PERF);
	if (ret != AI_SUCCESS) {
		std::cout << LOG_TAG << " load model fail " << std::endl;
		return -1;
	}
	struct timeval loadmodelend;
	gettimeofday(&loadmodelend, nullptr);
	double elapsedload = (loadmodelend.tv_sec - loadmodel.tv_sec) * 1000.0 +
						(loadmodelend.tv_usec - loadmodel.tv_usec) / 1000.0;
	std::cout << LOG_TAG << " Elapsed_Time for LoadModel: " << elapsedload
				<< std::endl;
	// 3. input image files/buffers;
	// DataFormat dataformat;
	InitDataFormat(&dataformat);
	dataformat.input_type = AISDK_FLOAT32;
	dataformat.output_type = AISDK_FLOAT32;
    dataformat.input_node_count = input_count;
	dataformat.output_node_count = output_count;
	inputbufs = reinterpret_cast<void**>(malloc(sizeof(float*)*input_count));
    NodeShape** innodes=reinterpret_cast<NodeShape**>(malloc(sizeof(NodeShape*)*input_count));
	for(int i=0;i<input_count;++i){
		NodeShape* node =
				reinterpret_cast<NodeShape*>(malloc(sizeof(NodeShape)));
		node->node_dim_size = input_shapes[i].size();
		node->node_name = const_cast<char *>(input_node_name[i].c_str());
		size_t inputsize=1;
		for(int j=0;j<(input_shapes[i]).size();j++){
            node->node_shape[j] = input_shapes[i][j];
			inputsize*=input_shapes[i][j];
		}
		inputbufs[i] = reinterpret_cast<void*>(malloc(inputsize * sizeof(float)));
		innodes[i]=node;
		std::cout << LOG_TAG << "  i  "<< " input node name: "
				<< input_node_name[i] << std::endl;
	}
	dataformat.input_nodes = innodes;
	NodeShape** outnodes = reinterpret_cast<NodeShape**>(malloc(sizeof(NodeShape*)*output_count));
	outputbufs=reinterpret_cast<void**>(malloc(sizeof(float)*output_count));
	for(int i=0;i<output_count;++i){
	    NodeShape* node = reinterpret_cast<NodeShape*>(malloc(sizeof(NodeShape)));
		node->node_dim_size = output_shapes[i].size();
		node->node_name = const_cast<char *>(output_node_name[i].c_str());
		size_t outputsize=1;
		for(int j=0;j<(output_shapes[i]).size();j++){
            node->node_shape[j] = output_shapes[i][j];
			outputsize*=output_shapes[i][j];
		}
		outputbufs[i] = reinterpret_cast<void*>(malloc(outputsize * sizeof(float)));
		outnodes[i]=node;
		std::cout << LOG_TAG  <<"  i  " << " output node name: "
					<< output_node_name[i] << std::endl;
		
	}
	dataformat.output_nodes = outnodes;
}


int main(int argc, char* argv[]) {
	signal( SIGINT, sig_handler);
	int ret=aisdk_init(argc,argv);
	if(ret==-1){
		std::cout << "ERROR: init !" << std::endl;
		return -1;
	}
	if (ReadLabelsFile(cli.labelfile(), &vec_labels_, &label_count_) != 0) {
		std::cout << "ERROR: read label file!" << std::endl;
	}
	//init grid_strides
	DIR *dp;
	struct dirent *dirp;

    cv::VideoCapture capture;
    if (!read_mode)
    { // read VideoCapture or
        capture.open(video_number);
        while (!capture.isOpened() && keepRunning)
        {
            std::cout << "VideoCapture is not open!!!" << std::endl;
            capture.open(video_number);
            sleep(1);
            std::cout <<  "VideoCapture Reloading"<<std::endl;
        }
        capture.set(3,OUTPUT_SIZE_W);
		capture.set(4,OUTPUT_SIZE_H);
    }
    else { //read local images
		if((dp = opendir(image_folder.c_str())) == NULL){
			std::cout << LOG_TAG << " Opendir failed : "<< image_folder << std::endl;
			return -1; //open error
		}
	}

	while( keepRunning)
	{
		struct timeval post;
		gettimeofday(&post, nullptr);

        if (!read_mode)
        { // read capture
            capture >> imgbase;
        }
        else{ //read image folder
			if((dirp = readdir(dp)) == NULL){
                std::cout << "image folder: " << image_folder << " read done!"  << std::endl;
                keepRunning=0;
                continue;
			}
			std::string img_path;
			if(strcmp(dirp->d_name,".")&&strcmp(dirp->d_name,"..")){
                img_path = image_folder+"/"+dirp->d_name;
                std::cout << "read image path:" << img_path << std::endl;
				file_name=std::string(dirp->d_name);
            }
            else{
			    continue;
			}
			imgbase=cv::imread(img_path);
		}
		
		if(imgbase.empty()){
			std::cout <<  "image is empty!!!"<<std::endl;
			continue;
		}
        // cv::resize(imgbase, imgbase, cv::Size(INPUT_SIZE, INPUT_SIZE)); 
		img_w = imgbase.size().width;
		img_h = imgbase.size().height;
		//yolox just one input and one output
		get_img<float>(reinterpret_cast<float*>(inputbufs[0]), input_shapes[0], false, false);
		RunModel(modelManager, inputbufs, input_count, &dataformat,outputbufs, output_count, 1000);
		std::cout << "RunModel done" << std::endl;

		select_bbox(outputbufs,img_w,img_h,img_scale);

		struct timeval postend;
		gettimeofday(&postend, nullptr);
		double elapsedpost = (postend.tv_sec - post.tv_sec) * 1000.0 +
					(postend.tv_usec - post.tv_usec) / 1000.0;
		std::cout << LOG_TAG << " Elapsed time for Endpost processing: " << elapsedpost << "ms" << std::endl;
		show_rect();
        std::cout << LOG_TAG << " Elapsed time for Endpost processing: " << elapsedpost << "ms" << std::endl;
        printf("imshow\n");
        if(show_flow>=0){
            cv::imshow("result", imgbase);
            
            auto k=cv::waitKey(show_flow);
            if(k>='a' && k <= 'z'){ //Press any lowercase letter to pause
                cv::waitKey(0);
            }
        }
        static int i=0;
        static char name[50];
        //sprintf(name,"output/output_%d.jpg",i++); 
        //sprintf(name, "output/%s", file_name.c_str());
        //cv::imwrite("imgbase",imgbase);
        cv::imshow("imgbase",imgbase);
        cv::waitKey(10);
        
	}
	free_2d_arr(dataformat.input_nodes,input_count);
	free_2d_arr(dataformat.output_nodes,output_count);
	free_2d_arr(inputbufs,input_count);
	free_2d_arr(outputbufs,output_count);
	DestroyModelManager(modelManager);
	if(!read_mode)capture.release();
	printf("\nrelese over!\n");
	return 0;
}
