void onMouse(int event, int x, int y, int flags, void* param);//鼠标事件
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)                    //颜色变换
cv::Mat stereo_match_sgbm(int, void*)                                         //SGBM匹配算法
void onMouse(int event, int x, int y, int flags, void* param)  //evnet:鼠标事件类型 x,y:鼠标坐标 flags：鼠标哪个键