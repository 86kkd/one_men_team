//g++ `pkg-config --cflags opencv` sgbm640.cpp `pkg-config --libs opencv` -o test.out
//sudo ./test.out
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


void onMouse(int event, int x, int y, int flags, void* param);//鼠标事件

const int imageWidth = 640;                      //请参考相机参数进行修改
const int imageHeight = 400;                      //请参考相机参数进行修改

cv::Mat frame, frame_L, frame_R;
cv::Size imageSize = cv::Size(imageWidth, imageHeight);
uchar HNY_CV_002 = 0;
cv::Mat rgbImageL, grayImageL;
cv::Mat rgbImageR, grayImageR;
cv::Mat rectifyImageL, rectifyImageR;

cv::Rect validROIL;                                   //图像校正之后，会对图像进行裁剪，这里的validROI指裁剪之后的区域  
cv::Rect validROIR;

cv::Mat mapLx, mapLy, mapRx, mapRy;                   //映射表  
cv::Mat Rl, Rr, Pl, Pr, Q;                            //校正旋转矩阵R，投影矩阵P, 重投影矩阵Q
cv::Mat xyz;                                          //三维坐标
const double f=784.41024; // mm
double base_line=60.12119;//mm
cv::Point origin;                                     //鼠标按下的起始点
cv::Rect selection;                                   //定义矩形选框
bool selectObject = false;                        //是否选择对象
cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) << 447.7175, 0,       319.9630,
	                                       0,       447.2938, 199.3117,
	                                       0,       0,       1);

cv::Mat distCoeffL = (cv::Mat_<double>(5, 1) << -0.11524,   0.020997,   -0.04, 0, 0);
//[kc_left_01,  kc_left_02,  kc_left_03,  kc_left_04,   kc_left_05]


/*右目相机标定参数------------------------
fc_right_x   0              cc_right_x
0            fc_right_y     cc_right_y
0            0              1
-----------------------------------------*/
cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << 446.5157, 0,       328.1410,
	                                       0,       446.3321, 193.28034,
	                                       0,       0,       1);

cv::Mat distCoeffR = (cv::Mat_<double>(5, 1) << -0.0085,   -0.0209,   0.00548,  0, 0);
//[kc_right_01,  kc_right_02,  kc_right_03,  kc_right_04,   kc_right_05]


cv::Mat T = (cv::Mat_<double>(3, 1) << -59.8235,     0.27450,          0.527668);     //T平移向量
							 //[T_01,        T_02,       T_03]

cv::Mat R = (cv::Mat_<double>(3,3)<<
	0.999, -0.0005, 0.0035,
	0.0005, 0.9999, -0.003,
	-0.0035, 0.003, 0.9999);

// cv::Mat rec = (cv::Mat_<double>(3, 1) << -0.00002,   -0.00125,    0.00032);   //rec旋转向量
// 							  //[rec_01,     rec_02,     rec_03]
//########--双目标定参数填写完毕-----------------------------------------------------------------------


// cv::Mat R;                                              
//--------------------------------------------------------------------------------------------------------




void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)                    //颜色变换
{ 
	float max_val = 255.0f;
	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
	float sum = 0;
	for (int i = 0; i<8; i++)
		sum += map[i][3];

	float weights[8];   
	float cumsum[8];  
	cumsum[0] = 0;
	for (int i = 0; i<7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	int height_ = src.rows;
	int width_ = src.cols;
 
	for (int v = 0; v<height_; v++) {
		for (int u = 0; u<width_; u++) {
 
			float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

			int i;
			for (i = 0; i<7; i++)
				if (val<cumsum[i + 1])
					break;
 
			float   w = 1.0 - (val - cumsum[i])*weights[i];
			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
			 
			disp.data[v*width_ * 3 + 3 * u + 0] = b;                       //rgb内存连续存放 
			disp.data[v*width_ * 3 + 3 * u + 1] = g;
			disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}
cv::Mat stereo_match_sgbm(int, void*)                                         //SGBM匹配算法
{

	//【可能需要修改的程序参数4】：以下为SGBM算法重要的3个参数
	int min_disparity = 1;                                                 //最小视差
    int block_size = 3;                                                //窗口的大小
	int num_disparity = 16*8;                                                //最大的视差，要被16整除

	cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(min_disparity, num_disparity, block_size);

	int P1 = 8 * rectifyImageL.channels() * block_size* block_size;  //惩罚系数1
	int P2 = 128 * rectifyImageL.channels() * block_size* block_size; //惩罚系数2



	left_matcher->setP1(P1);
	left_matcher->setP2(P2);

	left_matcher->setPreFilterCap(63);                                             //滤波系数
	left_matcher->setUniquenessRatio(10);                                          //代价方程概率因子
	left_matcher->setSpeckleRange(1);                                              //相邻像素点的视差值浮动范围
	left_matcher->setSpeckleWindowSize(0);                                       //针对散斑滤波的窗口大小
	left_matcher->setDisp12MaxDiff(1);                                             //视差图的像素点检查
	left_matcher->setMode(cv::StereoSGBM::MODE_HH);  



	cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
	auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
	
	// wlsFilter->setLRCthresh(24);
	// wlsFilter->setDepthDiscontinuityRadius(3);

	cv::Mat left_disp, right_disp;
	struct timeval time1 = {0, 0};
	struct timeval time2 = {0, 0};

	left_matcher->compute(rectifyImageL, rectifyImageR, left_disp);
	right_matcher->compute(rectifyImageR, rectifyImageL, right_disp);

	wlsFilter->setLambda(8000.0);  // 平滑项的强度
	wlsFilter->setSigmaColor(1.5); // 颜色相似性权重的标准差
	wlsFilter->setLRCthresh(24);

	wlsFilter->filter(left_disp,grayImageL,left_disp,right_disp);
	// cv::normalize(filtered_disp, left_disp, 0, 255, cv::NORM_MINMAX, CV_8U);
	// auto ROI = computeROI()
	// wlsFilter->filter(disp,right_disp,cv::Mat(),); 
	double elapsedrun = (time2.tv_sec - time1.tv_sec) * 1000.0 +
						(time2.tv_usec - time1.tv_usec) / 1000.0;
// std::cout << "compute time" << elapsedrun  << std::endl;
	cv::Mat disp8U = cv::Mat(left_disp.rows, left_disp.cols, CV_8UC1);                       //用于显示  

	cv::reprojectImageTo3D(left_disp, xyz, Q, true);                                //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16
	xyz = xyz * 16;
	

	left_disp.convertTo(left_disp, CV_32F, 1.0 / 16);                                //除以16得到真实视差值
	
	cv::Mat depth_map(left_disp.rows, left_disp.cols, CV_16SC1);//深度图
	// float *p_depth=depth_map.data;
	//CV_8UC1
	//CV_16SC1
	cv::Mat depth_map8U = cv::Mat(left_disp.rows, left_disp.cols, CV_8UC1);   

	for (int v = 0; v<left_disp.rows; v++) {
		for (int u = 0; u< left_disp.cols; u++){
				float disp_data=left_disp.at<float>(v,u);
				if(abs(disp_data)<0.1&&disp_data!=0){
					depth_map.at<int16_t>(v,u)=INT16_MAX;
					continue;
				}
				depth_map.at<int16_t>(v,u)=(int16_t)(f*base_line/(float)(disp_data));
		}
	}
	cv::normalize(left_disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(depth_map, depth_map8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// medianBlur(disp8U, disp8U, 5);                                         //中值滤波
	// medianBlur(depth_map8U, depth_map8U, 5);                                         //中值滤波
	//显示视差图
	cv::Mat dispcolor(disp8U.size(), CV_8UC3);
	GenerateFalseMap(disp8U, dispcolor);
	// cv::Mat filteredDepthMap;  // 输出滤波后的深度图
	// wlsFilter.filter(dispcolor, inputImage, filteredDepthMap);

	cv::imshow("disparity", dispcolor);
	// cv::imshow("depth_map", depth_map8U);
	// cv::setMouseCallback("depth_map", onMouse, reinterpret_cast<void*>(&depth_map)); //关联图像显示窗口和onMouse函数  滤波对导致显示的视差图和深度图不对应，所以去掉了滤波
	cv::setMouseCallback("disparity", onMouse, reinterpret_cast<void*>(&depth_map)); //关联图像显示窗口和onMouse函数
	if(cv::waitKey(1)=='t'){
		cv::waitKey(0); 
	}
	return depth_map;
}
void onMouse(int event, int x, int y, int flags, void* param)  //evnet:鼠标事件类型 x,y:鼠标坐标 flags：鼠标哪个键
{
	cv::Mat* im = reinterpret_cast<cv::Mat*>(param);
	switch (event) {

	case cv::EVENT_LBUTTONDOWN:
		//显示图像像素值

		if (static_cast<int>(im->channels()) == 1)
		{
			//若图像为灰度图像，则显示鼠标点击的坐标以及灰度值
			int16_t d= im->at<int16_t>(cv::Point(x, y));
			std::cout << "at (" << x << ", " << y << " ) depth is: " <<d << " mm"<< std::endl;
		}
		else
		{
			//若图像为彩色图像，则显示鼠标点击坐标以及对应的B, G, R值
			std::cout << "at (" << x << ", " << y << ")"
				<< "  B value is: " << static_cast<int>(im->at<cv::Vec3b>(cv::Point(x, y))[0]) 
				<< "  G value is: " << static_cast<int>(im->at<cv::Vec3b>(cv::Point(x, y))[1])
				<< "  R value is: " << static_cast<int>(im->at<cv::Vec3b>(cv::Point(x, y))[2])
				<< std::endl;
		}

		break;
	}
}
int main(){

    //--立体校正-------------------------------------------------------------------
	// cv::Rodrigues(rec, R);                                   //Rodrigues变换
	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
    
    cv::VideoCapture cap;

	cap.open("/dev/video0");    //【可能需要修改的程序参数4】：打开端口1对应的设备，电脑自带摄像头一般编号为0，外接摄像头编号为1，也可能是反过来的

	cap.set(cv::CAP_PROP_FRAME_WIDTH, imageWidth * 2); //设置捕获图像的宽度，为双目图像的宽度
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, imageHeight);  //设置捕获图像的高度
    
    cv::Mat canvas;
	double sf;
	int w, h;
	struct timeval time1 = {0, 0};
	struct timeval time2 = {0, 0};
    while (1)
	{
		//key = waitKey(1);                                 //获取键盘按下后的键值

		cap >> frame;

        frame_L = frame(cv::Rect(0, 0, imageWidth, imageHeight));             //获取左Camera的图像
        frame_R = frame(cv::Rect(imageWidth, 0, imageWidth, imageHeight));   //获取右Camera的图像

		//rgbImageL = imread("left11.bmp", CV_LOAD_IMAGE_COLOR);
		//rgbImageR = imread("right11.bmp", CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(frame_L, grayImageL, cv::COLOR_BGR2GRAY);
		cv::cvtColor(frame_R, grayImageR, cv::COLOR_BGR2GRAY);
		/*namedWindow("ImageL Before Rectify", WINDOW_NORMAL);  imshow("ImageL Before Rectify", grayImageL);
		namedWindow("ImageR Before Rectify", WINDOW_NORMAL);  imshow("ImageR Before Rectify", grayImageR);*/
		//--经过remap之后，左右相机的图像已经共面并且行对准----------------------------------------------
               	
		cv::remap(grayImageL, rectifyImageL, mapLx, mapLy, cv::INTER_LINEAR);
		cv::remap(grayImageR, rectifyImageR, mapRx, mapRy, cv::INTER_LINEAR);


		//--把校正结果显示出来---------------------------------------------------------------------------

			/*cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);
			cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);*/
			//imwrite("rectifyImageL.jpg", rectifyImageL);imwrite("rectifyImageR.jpg", rectifyImageR);

			//namedWindow("ImageL After Rectify", WINDOW_NORMAL); imshow("ImageL After Rectify",/* rgbRectifyImageL);
			//namedWindow("ImageR After Rectify", WINDOW_NORMAL); imshow("ImageR After Rectify", */rgbRectifyImageR);


		//--显示在同一张图上-----------------------------------------------------------------------------

		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);
		h = cvRound(imageSize.height * sf);
		canvas.create(h, w * 2, CV_8UC3);
		// cv::namedWindow("frame_L", cv::WINDOW_NORMAL);  
		cv::namedWindow("frame_L");  
		cv::imshow("frame_L", frame_L);

		//--显示结果-------------------------------------------------------------------------------------
			//namedWindow("disparity", WINDOW_NORMAL);

		//--鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)------------
			//setMouseCallback("disparity", onMouse, 0);
		gettimeofday(&time1, nullptr);//此时获取当前系统时间
		stereo_match_sgbm(0, 0);   //运行sgbm算法
		gettimeofday(&time2, nullptr);//此时获取当前系统时间
		double elapsedrun = (time2.tv_sec - time1.tv_sec) * 1000.0 +
		              (time2.tv_usec - time1.tv_usec) / 1000.0;
// std::cout << "done time" << elapsedrun  << std::endl;
		// cv::waitKey(1);
	}
    return 0;
}
