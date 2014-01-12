 
//  Process.cpp
//  LaneDetectorSim1.1
//
//  Created by LI XUANPENG on 09/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//
//  detectLaneFlag:    (init) -1  --  0 (normal)
//                             |     ||
//                  (changing) 2  --  1 (possible)
//                                     

#include "Process_LaneDetectorSim.h"

extern const int    WIDTH;
extern const int    HEIGHT;
extern const double COEF;
extern const double YAW_ANGLE;

extern const int    FRAME_START;

extern const int    TH_KALMANFILTER;
extern const int    TH_LANECHANGE;

extern const int    WIN_COLS;
extern const int    WIN_ROWS;



namespace LaneDetectorSim{    
    /********************************************************************************/
    //This function processes the video and detects the lanes
    //! Record the effected data after Kalman tracking works.
    //! When lane changes, we reset the Kalman filter and laneKalmanIdx.
    //! Considering the situation that the lane changes but Kalman filter not works
    /********************************************************************************/
    void ProcessLaneImage(cv::Mat &laneMat, 
                          const LaneDetector::LaneDetectorConf &laneDetectorConf, 
                          const double &startTime,
                          cv::KalmanFilter &laneKalmanFilter, 
                          cv::Mat &laneKalmanMeasureMat, int &laneKalmanIdx, 
                          std::vector<cv::Vec2f> &hfLanes, 
                          std::vector<cv::Vec2f> &lastHfLanes,
                          double &lastLateralOffset,
                          double &lateralOffset, int &isChangeLane,
                          int &detectLaneFlag,  const int &idx, double &execTime, 
                          std::vector<cv::Vec2f> &postHfLanes, int &changeDone)
    {   
        char *text = new char[100];
        
        std::cout << std::endl;
        std::cout << "---" << idx << std::endl;
//        std::cout << "laneKalman measure before" << std::endl;
//        LaneDetector::PrintMat(laneKalmanMeasureMat);
        
        //! Reduce the size of raw image
        cv::resize(laneMat, laneMat, cv::Size(cvRound(WIDTH * COEF), cvRound(HEIGHT * COEF)), cv::INTER_AREA);
        //! Change color to grayscale                
        cv::Mat grayMat = cv::Mat(cvRound(HEIGHT * COEF), cvRound(WIDTH * COEF), CV_8UC1);
        cv::cvtColor(laneMat, grayMat, cv::COLOR_BGR2GRAY); 
        
        std::vector<LaneDetector::Lane> lanes, postLanes;
        cv::Point2d vanishPt, leftCornerPt, rightCornerPt;
        const int offsetY = cvRound(laneMat.rows * YAW_ANGLE);
        
        std::cout << "--- isChangeLane : " << isChangeLane << std::endl;
        
        /************************************************************/
        //! Detect lanes on the original image.
        /************************************************************/
        hfLanes.clear();
        //! Run in the same conditions as DetectLanes
        //! The same in the next frame
        //! draw the predicted region when laneKalmanIdx exceeds threshold
        LaneDetector::DrawPreROI(laneMat, postHfLanes, laneKalmanIdx, isChangeLane);
        
        DetectLanes(grayMat, laneDetectorConf, offsetY, hfLanes, postHfLanes, laneKalmanIdx, isChangeLane);
        
        //! Draw the detected lanes
        if (!hfLanes.empty()) 
        {
            hfLanetoLane(laneMat, hfLanes, lanes);
            
            for (std::vector<LaneDetector::Lane>::const_iterator iter = lanes.begin(); iter != lanes.end(); ++iter) {
//                circle(laneMat, iter->startPoint, 5, CV_RGB(255,0,0), 2);
//                circle(laneMat, iter->endPoint, 5, CV_RGB(255,0,0), 2);
                cv::line(laneMat, cv::Point2d(iter->startPoint.x, iter->startPoint.y+offsetY), cv::Point2d(iter->endPoint.x, iter->endPoint.y+offsetY), CV_RGB(255, 255, 0), 3);
            }
            
            if(hfLanes.size() == 2) {
//                std::cout << "Update LO in Detection" << std::endl;
                LaneDetector::GetMarkerPoints(laneMat, hfLanes, vanishPt, leftCornerPt, rightCornerPt, offsetY);
                LaneDetector::GetLateralOffset(laneMat, leftCornerPt.x, rightCornerPt.x, lateralOffset);
                std::cout << "Detection >> LO: "<< lateralOffset << std::endl; 
            }
        }
        
        int initDone = 0;
        //! Init finished and it can detect 2 lanes, only run in the init step
        if (detectLaneFlag == -1 && hfLanes.size() == 2) {
            detectLaneFlag = 0;//init state -1 -> normal state 0
            initDone = 1;
        }//else continue to init 
        
        //! Consider the next frame lost after the first successful init frame
//        if(detectLaneFlag == -1 && hfLanes.size() == 2)
//        {
//            laneKalmanMeasureMat.at<float>(0) = (float)hfLanes[0][0];
//            laneKalmanMeasureMat.at<float>(1) = (float)hfLanes[0][1]; 
//            laneKalmanMeasureMat.at<float>(2) = 0;
//            laneKalmanMeasureMat.at<float>(3) = 0;
//            laneKalmanMeasureMat.at<float>(4) = (float)hfLanes[1][0];
//            laneKalmanMeasureMat.at<float>(5) = (float)hfLanes[1][1];
//            laneKalmanMeasureMat.at<float>(6) = 0;
//            laneKalmanMeasureMat.at<float>(7) = 0;
//        }
        
        if(detectLaneFlag == 2 && hfLanes.size() == 2 && laneKalmanIdx == 0)
        {
            std::cout << "Lane Change Done!" << std::endl;
            changeDone = 1;
        }
//        std::cout << "Change Done Flag : " << changeDone  << std::endl;
        /********************************************************/
        //! 1.Predict Lane Regions with Kalman Filter
        //! 2.Track the Lanes
        /********************************************************/
        /* 
         * @condition 1 : detectLaneFlag == 0 && initDone == 0 
         *              -> No tracking in first successful initial frame  
         * @condition 2 : detectLaneFlag == 1 && isChangeLane != -1 && isChangeLane != 1
         *              -> No tracking in the lane departure out of limitation
         * @condition 3 : detectLaneFlag == 2 && changeDone == 1
         *              -> No tracking in first lane change without init
         */
        std::vector<cv::Vec2f> preHfLanes;
        postHfLanes.clear();
        
        if( (detectLaneFlag == 0 && initDone == 0) 
           || (detectLaneFlag == 1 && isChangeLane != -1 && isChangeLane != 1) 
           || (detectLaneFlag == 2 && changeDone == 1) ) 
        {
            
            LaneDetector::TrackLanes_KF(grayMat, laneKalmanFilter, laneKalmanMeasureMat, hfLanes, lastHfLanes, preHfLanes, postHfLanes);
            
            laneKalmanIdx++;

            if ( !postHfLanes.empty() 
                && laneKalmanIdx >= TH_KALMANFILTER ) 
            {
                hfLanetoLane(laneMat, postHfLanes, postLanes);
                for (std::vector<LaneDetector::Lane>::const_iterator iter = postLanes.begin(); iter != postLanes.end(); ++iter) 
                {
                    line(laneMat, cv::Point2d(iter->startPoint.x, iter->startPoint.y+offsetY), cv::Point2d(iter->endPoint.x, iter->endPoint.y+offsetY), CV_RGB(0, 255, 0), 3);
                }
                
                if( hfLanes.size() != 2 
                   || std::abs(lateralOffset - lastLateralOffset) > 0.1 ) 
                {
                    LaneDetector::GetMarkerPoints(laneMat, postHfLanes, vanishPt, leftCornerPt, rightCornerPt, offsetY);
                    LaneDetector::GetLateralOffset(laneMat, leftCornerPt.x, rightCornerPt.x, lateralOffset);
                    LaneDetector::DrawMarker(laneMat, postHfLanes, lateralOffset);
                
                    sprintf(text, "Data From Tracking");
                    cv::putText(laneMat, text, cv::Point(laneMat.rows/2, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0));
                }
                else {
                    LaneDetector::DrawMarker(laneMat, hfLanes, lateralOffset);
                    
                    sprintf(text, "Data From Detection");
                    cv::putText(laneMat, text, cv::Point(laneMat.rows/2, 20),cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 0, 255));
                }
            }
        }

        //! After use lastHfLanes: hfLanes(k-1), update it.
        lastHfLanes = hfLanes; 
        
        /********************************************************/
        //! Change Lane Detect Flag
        //! When init finished in current frame,
        //! the process will not come into lane tracking
        /********************************************************/
        //! \param lateralOffset updates to current one
        //! Lane change happened, lateralOffset is the last lateralOffset here
        if ( detectLaneFlag == 0 && std::abs(lateralOffset) == 1 )
        {
            detectLaneFlag = 1; // Changing lane may happen
        } 
        else if( detectLaneFlag == 1 && std::abs(lateralOffset) == 1 ) 
        {
            detectLaneFlag = 0; // Return to the original lanes 
        } //else keep the last state
        std::cout << "detectLaneFlag: " << detectLaneFlag << std::endl;

        
        /********************************************************/
        //! Operation in different Lane Detect Flag 
        //! If init finished and it can correctly track lanes
        /********************************************************/
        if (detectLaneFlag == 0 && initDone == 0) 
        {
            isChangeLane = 0;
        }
        else if (detectLaneFlag == 1) 
        {
            //! If Lane changes in last frame and lane can be tracked in current frame.
            //! In this condition, lateralOffset = 1 from the tracked lane
            isChangeLane = 0;
        
            
            //! If lateral deviation exceeds more than an entire vehcile, 
            //! it would reset Kalman filter
            int departCond = 0; //not entire outside
            const double OutTh = 0.3;
            if(!postHfLanes.empty())
            {
                std::cout << "postHfLanes >> theta_L: " << postHfLanes[0][1] << " theta_R: " << postHfLanes[1][1] << std::endl;
                if(lateralOffset < 0)//car to left
                {
                    if (postHfLanes[0][1] < OutTh) {
                        //left side is out of left lane marking
                        departCond = 1;
                    }else {
                        // otherwise
                        departCond = 0;
                    }
                } 
                else //car to right
                { 
                    if(postHfLanes[1][1] > -OutTh) {
                        departCond = 1;
                    } else {
                        departCond = 0;
                    }
                }
            }
            
            if(departCond == 1)
            {
                std::cout << "Changing Lanes >> Reset All Parameters" << std::endl;
                //! Reset Kalman filter and laneKalmanIdx = 0
                LaneDetector::InitLaneKalmanFilter(laneKalmanFilter, laneKalmanMeasureMat, laneKalmanIdx);
                postHfLanes.clear();
                
                //! Next detection will occur in the whole image, NOT ROI
                if(lateralOffset == -1)
                    isChangeLane = -1;//car departs towards left
                else
                    isChangeLane = 1;//car departs towards right

                detectLaneFlag = 2;
                lastHfLanes.clear();
            }
        }
        else if(detectLaneFlag == 2 && changeDone == 1)
        {
            //! If Lane changes in last frame and lane can be tracked in current frame.
            //! In this condition, lateralOffset = 1 from the tracked lane
            if(std::abs(lateralOffset) < 1) {
                detectLaneFlag = 0;
                changeDone = 0;
            }
            if(laneKalmanIdx > TH_KALMANFILTER)
                isChangeLane = 0;
        }
        
        /********************************************************/
        //! Check whether the tracked lane can fit to the real situation
        /********************************************************/
        const double DisTH = 20;
        const int    HeightTh = cvRound(laneMat.rows / 3.0); // This Value is high related with Vanishing Point
        int isTrackFailed = 0;
        
        if(!postHfLanes.empty())
        {
            LaneDetector::GetMarkerPoints(laneMat, postHfLanes, vanishPt, leftCornerPt, rightCornerPt, offsetY);
            cv::circle(laneMat, vanishPt, 4, CV_RGB(255, 0 , 255), 2);
            
            //! If tracking information wrong, 
            //! it should reset the Kalman filter and re-detect the lanes.
            if(vanishPt.y > HeightTh || std::abs(leftCornerPt.x - rightCornerPt.x) < DisTH) 
            {
                LaneDetector::InitLaneKalmanFilter(laneKalmanFilter, laneKalmanMeasureMat, laneKalmanIdx);
                postHfLanes.clear();
                isTrackFailed = 1;
            }
        }

		if (std::isnan(lateralOffset))
			lateralOffset = lastLateralOffset;        

        if(!isTrackFailed)
        {
            lastLateralOffset = lateralOffset; //record last LO before LO updates
        } else {
            sprintf(text, "Fail in Tracking");
            cv::putText(laneMat, text, cv::Point(laneMat.rows/2, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0, 255, 0));
        }
        /********************************************************/
        //! Draw Lane Tracking Result on Raw Image
        //! Present running time of lane detection
        /********************************************************/
        execTime = ((double)cv::getTickCount() - startTime)/cv::getTickFrequency();
        
        //! Show index of frames
        sprintf(text, "Frame: %d", idx);
        cv::putText(laneMat, text, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0));
        //! Show the process time
        sprintf(text, "Process: %.2f Hz", 1.0/execTime);
        cv::putText(laneMat, text, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0));
//        //! Show Kalman index
//        sprintf(text, "KalmanIdx: %d", laneKalmanIdx);
//        cv::putText(laneMat, text, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0, 255, 0));
        delete text;
    }//end ProcessLaneImage
    

    void InitRecordData(std::ofstream &file, const char* fileName, const std::string *strName, const int &elemNum)
    {
        file.open(fileName);
        
        for(int i = 0; i < elemNum; i++ )
        {
            file <<  std::setiosflags(std::ios::fixed) << std::setw(15)  << strName[i];
        }
        file << std::endl;
    }
    
      
    void RecordLaneFeatures(std::ofstream &file, const LaneDetector::LaneFeature &laneFeatures, const double &pastTime)
    {
        file << std::setiosflags(std::ios::fixed) << std::setprecision(0) << std::setw(15) << laneFeatures.frame;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.lateralOffset;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LATSD;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LATSD_Baseline;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LATMEAN;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LATMEAN_Baseline;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LANEDEV;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LANEDEV_Baseline;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.LANEX;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.TLC;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(0) << std::setw(15) << laneFeatures.TLC_2s;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.TLCF_2s;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(0) << std::setw(15) << laneFeatures.TLC_halfs;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.TLCF_halfs;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5) << std::setw(15) << laneFeatures.TLC_min;
        file << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(15) << pastTime;
        file << std::endl;
    }
    
    void CodeMsg(const LaneDetector::LaneFeature &laneFeatures, char *str)
    {
        char *temp = new char[100];
        sprintf(temp, "%d", 0);//lane marker
        strcat(str, temp);
        sprintf(temp, ", %d", laneFeatures.frame);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.lateralOffset);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LATSD);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LATSD_Baseline);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LATMEAN);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LATMEAN_Baseline);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LANEDEV);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LANEDEV_Baseline);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.LANEX);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.TLC);
        strcat(str, temp);
        sprintf(temp, ", %d", laneFeatures.TLC_2s);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.TLCF_2s);
        strcat(str, temp);
        sprintf(temp, ", %d", laneFeatures.TLC_halfs);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.TLCF_halfs);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.TLC_min);
        strcat(str, temp);
        sprintf(temp, ", %f", laneFeatures.TOT);
        strcat(str, temp);
        delete temp;
    }

}//LaneDetectorSim
