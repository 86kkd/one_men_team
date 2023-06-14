template <class T>
void hwc_to_chw(T *dst, T *src, int height, int width, int channel, const float *mean_vals, const float *norm_vals);
template <class T>
void free_2d_arr(T **arr2d, int size)
template <class T>
void get_img(T *imgData_buffer,             // [out] input buffer
             const std::vector<int> &shape, // [in] input shape
             bool is_nhwc = false,          // [in] true: NHWC; false: NCHW
             bool is_bgr_mode = false);
int ReadLabelsFile(const std::string& file_name,
                   std::vector<std::string>* result,
                   size_t* found_label_count);
 void sig_handler( int sig );
static void get_bbox(float **outputs,float prob_threshold);
static void select_bbox(void **outputbuf,int nw,int nh,float img_scale);
static void show_rect(cv::Mat dapth_map = cv::Mat());
int aisdk_init(int argc, char* argv[]);