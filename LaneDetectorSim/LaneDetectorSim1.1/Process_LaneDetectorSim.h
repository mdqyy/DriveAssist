//
//  Process.h
//  LaneDetectorSim1.1
//
//  Created by LI XUANPENG on 09/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//

#ifndef LaneDetectorSim_Process_h
#define LaneDetectorSim_Process_h

#include "../../LaneDetector/LaneDetector1.1/DetectLanes.h"
#include "../../LaneDetector/LaneDetector1.1/TrackLanes.h"
#include "../../LaneDetector/LaneDetector1.1/LaneDetectorTools.h"
#include "../../LaneDetector/LaneDetector1.1/GenerateLaneIndicators.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <vector>
#include <iterator>
#include <opencv2/opencv.hpp>

//!IPC
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include "errno.h"

namespace LaneDetectorSim{
    const int NUM_LANE = 16;
    const std::string laneFeatureName[NUM_LANE] = 
    {
        "Frame", 
        "LO", "LATSD", "LATSD_b", 
        "LATMEAN", "LATMEAN_b", 
        "LANEDEV", "LANEDEV_b",
        "LANEX", 
        "TLC", "TLC_2s","TLCF_2s", "TLC_halfs", "TLCF_halfs", "TLC_min",
        "Time"
    };
    
    void ProcessLaneImage(cv::Mat &laneMat, 
                          const LaneDetector::LaneDetectorConf &laneDetectorConf, 
                          const double &startTime,
                          cv::KalmanFilter &laneKalmanFilter, 
                          cv::Mat &laneKalmanMeasureMat, int &laneKalmanIdx, 
                          std::vector<cv::Vec2f> &hfLanes, 
                          std::vector<cv::Vec2f> &lastHfLanes, 
                          double & lastLateralOffset,
                          double &lateralOffset, int &isChangeLane,
                          int &detectLaneFlag,  const int &idx, double &execTime, 
                          std::vector<cv::Vec2f> &preHfLanes, int &changeDone); 
   
    
    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum);
    
    void RecordLaneFeatures(std::ofstream &file, const LaneDetector::LaneFeature &laneFeatures, const double &pastTime);
    
    void CodeMsg( const LaneDetector::LaneFeature &laneFeatures, char *str);
}
#endif
