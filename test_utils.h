// File Name:test_utils.h
// Date:     2019-07-09
// Copyright (C) 2019 UNISOC Technologies Co.,Ltd. All Rights Reserved

#ifndef V2_SAMPLE_INCLUDE_TEST_UTILS_H_
#define V2_SAMPLE_INCLUDE_TEST_UTILS_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "tclap/CmdLine.h"

template <class Type>
Type stringToNum(const std::string& str) {
  istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}

class TestCli {
 private:
  // Main CLI object pointer declaration.
  TCLAP::CmdLine* cmd;

  TCLAP::SwitchArg* combomodelArg;
  TCLAP::ValueArg<std::string>* modelArg;
  TCLAP::ValueArg<std::string>* outArg;
  TCLAP::ValueArg<std::string>* fileorderArg;
  TCLAP::ValueArg<std::string>* inputformatArg;
  TCLAP::ValueArg<std::string>* outputformatArg;
  TCLAP::ValueArg<std::string>* labelArg;
  TCLAP::ValueArg<std::string>* intensorshape;
  TCLAP::ValueArg<std::string>* outtensorshape;

  TCLAP::ValueArg<unsigned int>* timeoutArg;
  TCLAP::ValueArg<unsigned int>* inputDimArg;
  TCLAP::ValueArg<unsigned int>* outputDimArg;

  TCLAP::SwitchArg* usebufferArg;
  TCLAP::SwitchArg* unpackedinArg;
  TCLAP::SwitchArg* unpackedoutArg;

  // input file list object pointer declaration.
  TCLAP::UnlabeledMultiArg<std::string>* infilesArg;

 public:
  // Constructor.
  TestCli() {
    // Main CLI object instantiation.
    cmd = new TCLAP::CmdLine("Command description message.", ' ', "0.1");
    // false is the default value
    combomodelArg = new TCLAP::SwitchArg("c", "combomodel",
                        "input a combo model or not", *cmd, false);
    // must have a model file
    modelArg = new TCLAP::ValueArg<std::string>(
        "m", "modelfile", "model file path", true, "", "std::string", *cmd);
    // current path is the default value
    outArg = new TCLAP::ValueArg<std::string>(
        "o", "outputpath", "output file path", false, ".", "std::string", *cmd);
    // image file order, rgb or bgr
    fileorderArg = new TCLAP::ValueArg<std::string>(
        "f", "fileorder", "input file order, rgb or bgr", false, "rgb",
        "std::string", *cmd);

    // input file format float32 or uint8_t
    inputformatArg =
        new TCLAP::ValueArg<std::string>("", "inputformat", "input file format",
                                         false, "float32", "std::string", *cmd);
    // output file format float32 or uint8_t
    outputformatArg = new TCLAP::ValueArg<std::string>(
        "", "outputformat", "output file format", false, "float32",
        "std::string", *cmd);
    // label file path
    labelArg = new TCLAP::ValueArg<std::string>(
        "l", "labelfile", "label file path", true, "", "std::string", *cmd);
    // 1,3,224,224 is the default value
    intensorshape =
        new TCLAP::ValueArg<std::string>("", "inshape", "input tensor shape",
                                         false, "1,3,224,224", "n,c,h,w", *cmd);
    // 1,1,1,1000 is the default value
    outtensorshape =
        new TCLAP::ValueArg<std::string>("", "outshape", "output tensor shape",
                                         false, "1,1000,1,1", "n,c,h,w", *cmd);
    // 1000 is the default value
    timeoutArg = new TCLAP::ValueArg<unsigned int>(
        "t", "timeout", "timeout value", false, 1000, "unsigned integer", *cmd);

    // input dim , 4 is the default value
    inputDimArg = new TCLAP::ValueArg<unsigned int>(
        "", "inputdim", "input dim value", false, 4, "unsigned integer", *cmd);
    // output dim ,4 is the default value
    outputDimArg =
        new TCLAP::ValueArg<unsigned int>("", "outputdim", "output dim value",
                                          false, 4, "unsigned integer", *cmd);
    // true is the default value
    usebufferArg = new TCLAP::SwitchArg("b", "usebuffer",
                                        "input/output use buffer", *cmd, true);
    // true is the default value
    unpackedinArg =
        new TCLAP::SwitchArg("", "unpackedin", "input unpacked", *cmd, true);
    // true is the default value
    unpackedoutArg =
        new TCLAP::SwitchArg("", "unpackedout", "output unpacked", *cmd, true);
    // input file list object instantiation.
    infilesArg = new TCLAP::UnlabeledMultiArg<std::string>(
        "infiles", "input image file name(s)", true, "input image file(s)",
        *cmd);
  }

  // Destructor.
  virtual ~TestCli() {
    // Main CLI object deletion.
    delete cmd;
    // input file list object deletion.
    delete infilesArg;
  }

  // Input argument list parsing function.
  void Parse(int argc, const char* const* argv) {
    try {
      // Parse the args.
      cmd->parse(argc, argv);

      std::stringstream inshapes(intensorshape->getValue());
      std::stringstream outshapes(outtensorshape->getValue());
      for (unsigned i = 0; i < inputdim(); ++i) {
        std::string insingleValueStr;
        getline(inshapes, insingleValueStr, ',');
        inshape[i] = stringToNum<unsigned int>(insingleValueStr);
      }
      for (unsigned i = 0; i < outputdim(); ++i) {
        std::string outsingleValueStr;
        getline(outshapes, outsingleValueStr, ',');
        outshape[i] = stringToNum<unsigned int>(outsingleValueStr);
      }
      /*
      std::cout << "UNISOC_AISDK inshape: " << inshape[0] << "," << inshape[1]
                << "," << inshape[2] << "," << inshape[3] << std::endl;
      std::cout << "UNISOC_AISDK outshape: " << outshape[0] << ","
                << outshape[1] << "," << outshape[2] << "," << outshape[3]
                << std::endl;
      */
    } catch (std::exception& e) {
      std::cout << "UNISOC_AISDK " << e.what() << std::endl;
      std::cout << "UNISOC_AISDK something error" << std::endl;
    }
  }

  unsigned int inshape[4];
  unsigned int outshape[4];

  // input file list getter function.
  bool combomodel() { return combomodelArg->getValue(); }
  const std::vector<std::string>& infiles() { return infilesArg->getValue(); }
  const std::string& modelfile() { return modelArg->getValue(); }
  const std::string& outputpath() { return outArg->getValue(); }
  const std::string& fileorder() { return fileorderArg->getValue(); }
  const std::string& inputformat() { return inputformatArg->getValue(); }
  const std::string& outputformat() { return outputformatArg->getValue(); }
  const std::string& labelfile() { return labelArg->getValue(); }
  const std::string& inshapes() { return intensorshape->getValue(); }
  const std::string& outshapes() { return outtensorshape->getValue(); }
  unsigned int timeout() { return timeoutArg->getValue(); }
  unsigned int inputdim() { return inputDimArg->getValue(); }
  unsigned int outputdim() { return outputDimArg->getValue(); }
  bool usebuffer() { return usebufferArg->getValue(); }
  bool unpackedin() { return unpackedinArg->getValue(); }
  bool unpackedout() { return unpackedoutArg->getValue(); }
};
#endif  // V2_SAMPLE_INCLUDE_TEST_UTILS_H_
