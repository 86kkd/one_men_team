// File Name:unisocai.h
// Date:     2019-07-09
// Copyright (C) 2019 UNISOC Technologies Co.,Ltd. All Rights Reserved

#ifndef V2_INCLUDE_UNISOCAI_H_
#define V2_INCLUDE_UNISOCAI_H_
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

typedef void ModelMgr;

// NPU priority mode: NPU support different priority,
// before RunModel, please choose a mode.
//
// HIGH_PERF: High priority mode.
// NORMAL_PERF: Normal priority mode.
// DEFAULT_PERF: Default priority mode.
enum NpuPref { HIGH_PERF, NORMAL_PERF, DEFAULT_PERF };

// NPU support data type
enum DataType {
  AISDK_NONE = 0,
  AISDK_Q8A = 1,
  AISDK_FLOAT32 = 2,
  AISDK_ERROR = -1
};

// Data Shape about the input or output node
// node_name: default is nullptr, only use pb need to assign a value.
// output_node_name: default is nullptr, only use pb need to assign a value.
typedef struct {
  char *node_name;
  unsigned int node_dim_size;
  unsigned int node_shape[4];
} NodeShape;

typedef struct {
  // is_nchw ：
  //     input shape is NCHW set true;
  //     input shape is NHWC set false;
  bool is_nchw;
  // need set both of input and output type, model input type maybe
  // different with output type, like SSD mobilenet v1/v2.
  DataType input_type;
  DataType output_type;
  // input data has packed for NNA or not, true is unpacked
  bool unpacked_in;
  // output data has packed by NNA or not, true is unpacked
  bool unpacked_out;
  unsigned int input_node_count;
  NodeShape **input_nodes;
  unsigned int output_node_count;
  NodeShape **output_nodes;
} DataFormat;

typedef struct {
  char *model_path;
  bool is_offline_model;
  DataFormat model_shape;
} MixModel;

typedef enum {
  VENDOR_MODEL_NONE = 0,
  IMGTEC_MODEL = 1,
  CAMB_MODEL = 2,
  VENDOR_MODEL_MAX = 3
} VendorModel;

enum UNISOCAIFramework {
  UNISOCAI_CAFFE_RESERVED,
  UNISOCAI_CAFFE2_RESERVED,
  UNISOCAI_TENSORFLOW,
  UNISOCAI_TFLITE_RESERVED
};

enum AIRet {
  /// common define
  // return fail when error
  AI_FAILED = -1,
  // All operations completed successfully
  AI_SUCCESS = 0,
  // Invalid parameters were provided
  AI_INVALID_PARAMS,
  // A timeout occurred while executing the network
  AI_TIMEOUT,
  // Out of memory occurred while allocating the memory or creating the session
  AI_OUT_OF_MEMORY,
  // Read model or input file permission denied
  AI_ACCES_ERROR,
  /// ModelManager stage
  // Uninitialized
  AI_INIT_ERROR,
  // No devices were found when creating the session
  AI_NO_DEVICE,
  // Device or resource busy
  AI_BUSY_ERROR,
  // Failure on create session
  AI_SESSION_ERROR,
  // Init DataFormat Error
  AI_INIT_DATAFORMAT_ERROR,
  // Init DataFormat sucess
  AI_INIT_DATAFORMAT_SUCESS,
  /// Load Model stage
  // Invalid model which NPU couldn't
  AI_INVALID_MODEL,
  // A Read error while reading the model
  AI_READ_MODEL_ERROR,
  // A generic error occurred while parsing the model file
  AI_MODEL_PARSE_ERROR,
  // An operation on the host layer returned an error
  AI_HOST_OPS_ERROR,
  // An unsupported operation was detected when parsing the model
  AI_UNSUPPORTED_OP_ERROR,
  // Invalid config file for mix model
  AI_INVALID_CONFIG_FILE,
  // load model error
  AI_LOAD_MODEL_ERROR,
  // load mix model error
  AI_LOAD_MIX_MODEL_ERROR,
  /// Load input file stage
  // Resource or file doesn't exist
  AI_NO_EXIST_ERROR,
  // Input buffer or file size was invalid
  AI_INVALID_SIZE_ERROR,
  // Input buffer was invalid
  AI_INBUF_POINTER_ERROR,
  // Input an invalid type
  AI_INVALID_TYPE_ERROR,
  /// Run Model stage
  // Failure on Stream
  AI_STREAM_ERROR,
  // Couldn't get output data
  AI_OUTPUT_ERROR,
  // Internal runtime logic detected an unexpected state
  AI_UNEXPECTED_STATE_ERROR,
  // Data transmission is broken
  AI_BROKEN_ERROR,
  // Failure to call runtime functions
  AI_FUNC_CALL_ERROR,
  // Unhandled error
  AI_UNHANDLED_ERROR,
  // Failure on event operation
  AI_EVENT_ERROR,
  // Failure on data reshape
  AI_RESHAPE_ERROR,
  // Invalid data descriptor
  AI_INVALID_DATADESC_ERROR,
  // Failure on run model
  AI_RUN_MODEL_ERROR,
  // Failure on run mix model
  AI_RUN_MIX_MODEL_ERROR,
  // new xmlparser error
  AI_INIT_XMLPARSER_ERROR,
  // new tensorflow impl error
  AI_INIT_TENSORFLOWIMPL_ERROR,
  // set cache error
  AI_SET_CACHE_ERROR,
  // set queue sieze error
  AI_SET_QUEUE_SIZE_ERROR,
  // nullprt error
  AI_NULLPTR_ERROR,
  // bad dim size
  AI_BAD_DIM_SIZE,
  // bad NPU version
  AI_BAD_NPU_VERSION,
  // malloc fail
  AI_MALLOC_FAILED,
  // unsupport function
  AI_UNSUPPORTED_FUNCTION,
  // NOTE: While adding a new error code, please add it above this line,
  // and you should modify the err_map(Api.cpp) at the same time
  // Unknown error
  AI_UNKNOWN_ERROR,
  // The last one
  AI_RET_MAX,
};

typedef enum {
  AISDK_LOG_ERROR = 1,
  AISDK_LOG_WARN = 2,
  AISDK_LOG_INFO = 3,
  AISDK_LOG_DEBUG = 4,
  AISDK_LOG_VERBOSE = 5,
} LogLevel;

// Return ModelMgr pointer
ModelMgr *CreateModelManager();

// Load model file,load success will return 0, load fail will return error code
//
// Param:
//     modelManager: pointer to ModelManager.
//     modelfile: pointer to model file path.
//     perf: set the mode of running model.
int LoadModel(ModelMgr *modelManager, const char *modelfile, NpuPref perf);

// Load mix model,load success will return 0, load fail will return error code
//
// Param:
//    modelManager: pointer to ModelManager.
//    configfile: the mix model config file.
//    perf: set the mode of running model.
int LoadMixModel(ModelMgr *modelManager, const char *configfile, NpuPref perf);

// initialize the DataFormat variable.
// Param:
//     dataformat: DataFormat variable.
//
//
int InitDataFormat(DataFormat *dataformat);

// Run Modle , run success will return 0, run fail will return error code
//
// Param:
//     modelManager: pointer to ModelManager.
//     infiles: input files.
//     filecount: infiles size
//     dataformat: input/output format, see @DataFormat define
//     outputpath: the output file save path.
//     timeout: set time out for inference(not support now).
// Note:
// 1. Make sure the dataformat meet case of model file
// which is used in LoadModel interface
int RunModel(ModelMgr *modelManager, const char *infiles[], int filecount,
             DataFormat *dataformat, const char *outputpath, const int timeout);

// Run Modle , run success will return 0, run fail will return error code
//
// Param:
//     modelManager: pointer to ModelManager.
//     inputbufs: input buffer.
//     inputcount: inputbufs size
//     dataformat: input/output format, see @DataFormat define
//     outputbufs: the output buffer.
//     ouputcount: outputbufs size
//     timeout: set time out for inference(not support now).
// Note:
// 1. Make sure the inputcount and outputcount meet case of model file
// which is used in LoadModel interface
// 2. Make sure the dataformat meet case of model file
// which is used in LoadModel interface
int RunModel(ModelMgr *modelManager, void *inputbufs[], int inputcount,
             DataFormat *dataformat, void *outputbufs[], int ouputcount,
             const int timeout);

// Run Mix Modle , run success will return 0, run fail will return error code
//
// Param:
//     modelManager: pointer to ModelManager.
//     inputbufs: input buffer.
//     inputcount: inputbufs size
//     outputbufs: the output buffer.
//     ouputcount: outputbufs size
//     timeout: set time out for inference (not support now).
// Note: Make sure the inputcount and outputcount meet case of model file
// which is used in LoadModel or LoadNixMoel interface
int RunMixModel(ModelMgr *modelManager, void *inputbufs[], int inputcount,
                void *outputbufs[], int ouputcount, const int timeout);

// Destory Model Manager,release res
//
// Param:
//     modelManager: pointer to ModelManager.
void DestroyModelManager(ModelMgr *modelManager);

// Set Log Level
//
// Param:
//     loglevel: the level of the log which will be set .
void SetLogLevel(LogLevel loglevel);

// Get sdk version
//
// Param:
//     sdkver: pointer to sdkver.
// note: the sdkver must have been alloced,and the length is longer than
// @GetSDKVersionLength()
void GetSDKVersion(char *sdkver);

// Get sdk version length
unsigned int GetSDKVersionLength();

#endif  // V2_INCLUDE_UNISOCAI_H_
