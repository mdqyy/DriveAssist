//
//  main.cpp
//  LaneDetectorSim1.1
//
//  Created by Xuanpeng Li on 09/13.
//  Copyright (c) 2013 ESIEE-AMIENS. All rights reserved.
//
#include "main_LaneDetectorSim.h"
#include "Process_LaneDetectorSim.h"      

#ifdef __cplusplus

/// Time
extern const double SAMPLING_TIME       = 60;   //sec for sampling lane features
extern const double SAMPLING_FREQ       = 8.4;  //Hz for camera
extern const double TIME_BASELINE       = 300;   //sec (300)
extern const int    NUM_WINDOW_EWM      = 5;    //EWMA, EWVAR Init (times)
/// Size of Image
extern const double COEF                = 0.75;
extern const int    WIDTH               = 640;
extern const int    HEIGHT              = 480;
extern const double YAW_ANGLE           = 0.1;
/// Multi-Image Show
extern const int    WIN_COLS            = 3;
extern const int    WIN_ROWS            = 3;

#ifdef __APPLE__
/// Record docs 
extern const char   FILE_LANE_FEATURE[] = "/Users/xuanpengli/Developer/Record/sim_laneFeature.txt";
/// Data Source
extern const char   LANE_RAW_NAME[]     = "/Users/xuanpengli/Developer/Data/LaneRaw_10-07-2013_18h30m21s/lane_%d.jpg";
extern const char   LANE_RECORD_NAME[]  = "/Users/xuanpengli/Desktop/lane_pic/lane_%d.png";
extern const char KEY_PATH[] 			= "/Users/xuanpengli/Developer/key.txt";
#elif defined __linux 
/// Record docs 
extern const char   FILE_LANE_FEATURE[] = "/home/lixp/Developer/Record/sim_laneFeature.txt";
/// Data Source
extern const char   LANE_RAW_NAME[]     = "/home/lixp/Developer/Data/LaneRaw_10-07-2013_18h30m21s/lane_%d.jpg";
extern const char   LANE_RECORD_NAME[]  = "/home/lixp/Developer/Record/IMAGE/lane_%d.png";
extern const char KEY_PATH[] 			= "/home/lixp/Developer/key.txt";
#endif

extern const int    TH_KALMANFILTER     = 1; //frames
extern const int    TH_LANECHANGE       = 10; // Considering as SAMPLING_FREQ

/// Frame 
extern const int    FRAME_START         = 1840;	 	//18:34:00 (1840)
extern const int    FRAME_STOP1         = 14007;    //18:57:59
extern const int    FRAME_RESTART1      = 15031;    //19:00:00
extern const int    FRAME_STOP2         = 17507;    //19:04:59
extern const int    FRAME_RESTART2      = 19497;    //19:09:00
extern const int    FRAME_END           = 20994; 	//19:11:59

/// Run applicaiton
extern const int    IMAGE_RECORD        = 1;

namespace LaneDetectorSim {
    
	int Process(int argc, const char* argv[])
	{
		int	LANE_DETECTOR 	= atoi(argv[1]);
		int	LANE_ANALYZER 	= atoi(argv[2]);
		int	SEND_DATA     	= atoi(argv[3]);
		int	DATA_RECORD   	= atoi(argv[4]); // Record data
		int	StartFrame		= atoi(argv[5]); // FRAME_START
		int EndFrame		= atoi(argv[6]); // FRAME_END

        int  idx            = StartFrame;  //index for image sequence
        int  sampleIdx      = 1;    //init sampling index
        char laneImg[100];
        
        
        std::ofstream laneFeatureFile;
        if (DATA_RECORD){
            InitRecordData(laneFeatureFile, FILE_LANE_FEATURE, laneFeatureName, NUM_LANE);
        }
        
/**********************************************************/
//Parameters for Lane Detector
/**********************************************************/
        cv::Mat laneMat;
        LaneDetector::LaneDetectorConf laneDetectorConf; 
        std::vector<cv::Vec2f> hfLanes;
        std::vector<cv::Vec2f> lastHfLanes;
        std::vector<cv::Vec2f> preHfLanes;
        //lane features
        std::vector<double> LATSDBaselineVec;
        std::deque<LaneDetector::InfoCar> lateralOffsetDeque;
        std::deque<LaneDetector::InfoCar> LANEXDeque;
        std::deque<LaneDetector::InfoTLC> TLCDeque;
        LaneDetector::LaneFeature laneFeatures;
        double lastLateralOffset = 0;
        double lateralOffset     = 0;    // Lateral Offset
        int    detectLaneFlag    = -1;   // init state -> normal state 0
        int    isChangeLane      = 0;    // Whether lane change happens
        int    changeDone        = 0;    // Finish lane change
        int    muWindowSize      = 5;    // Initial window size: 5 (sample)
        int    sigmaWindowSize   = 5;    // Initial window size: 5 (sample)
    
        // Initialize Lane Kalman Filter
        cv::KalmanFilter laneKalmanFilter(8, 8, 0);//(rho, theta, delta_rho, delta_theta)x2
        cv::Mat laneKalmanMeasureMat(8, 1, CV_32F, cv::Scalar::all(0));//(rho, theta, delta_rho, delta_theta)
        int    laneKalmanIdx     = 0;    //Marker of start kalmam

        if (LANE_DETECTOR) {
            LaneDetector::InitlaneDetectorConf(laneDetectorConf);
            
            LaneDetector::InitLaneKalmanFilter(laneKalmanFilter, laneKalmanMeasureMat, laneKalmanIdx);
        }
        
/**********************************************************/        
//! Inter-process communication
/**********************************************************/
        key_t ipckey;
        int mq_id;
        struct { 
            long type; 
            char text[1024]; 
        } laneMsg;

        if (SEND_DATA) 
        {
            /* Generate the ipc key */
            ipckey = ftok(KEY_PATH, 'a');
            if(ipckey == -1){
                printf("Key Error: %s\n", strerror(errno));
                exit(1);
            }
            
            mq_id = msgget(ipckey, 0);
            if (mq_id == -1) { 
                //MQ doesn't exit
                mq_id = msgget(ipckey, IPC_CREAT | IPC_EXCL | 0666);
                printf("LaneDetector creates a new MQ %d\n", mq_id);
            }
            else {
                //MQ does exit
                mq_id = msgget(ipckey, IPC_EXCL | 0666);
                printf("LaneDetector uses an existed MQ %d\n", mq_id);
            }
            //printf("Lane identifier is %d\n", mq_id);
            if(mq_id == -1) {  
                perror("error");
                _exit(1);  
            }
            //printf("This is the LaneDetectSim process, %d\n", getpid());
        }
        
/***************************************************/
//! Entrance of Process
//***************************************************/
        double initTime         = (double)cv::getTickCount();
        double intervalTime     = 0;
        double execTime         = 0;  // Execute Time for Each Frame
        double pastTime         = 0;
        double lastStartTime    = (double)cv::getTickCount();
        while (idx <= EndFrame)
        {
//            printf("\nProcess in %d frames: \n", idx);
            double startTime = (double)cv::getTickCount();
            /**********************************************************/
            //! Lane detect and tracking 
            /**********************************************************/ 
            if (LANE_DETECTOR)
            {
                sprintf(laneImg, LANE_RAW_NAME , idx);//match lane and face
                laneMat = cv::imread(laneImg);//imshow("laneMat", laneMat);
                
                ProcessLaneImage(laneMat, laneDetectorConf, startTime, laneKalmanFilter, laneKalmanMeasureMat, laneKalmanIdx, hfLanes, lastHfLanes, lastLateralOffset, lateralOffset, isChangeLane, detectLaneFlag,  idx, execTime, preHfLanes, changeDone);
            }
            
            /**********************************************************/  
            //! Calculate the running time for every sampling
            /**********************************************************/  
            //! past time
            pastTime = ((double)cv::getTickCount() - initTime)/cv::getTickFrequency();
//            printf("@Lane Sampling passes %f sec\n", pastTime);
            char *text_pastTime = new char[50];
            sprintf(text_pastTime, "Time: %.2f sec", pastTime);
            cv::putText(laneMat, text_pastTime, cv::Point(0, laneMat.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,255,0));
            delete text_pastTime;
            
            
            intervalTime = (startTime - lastStartTime)/ cv::getTickFrequency();//get the time between two continuous frames
            lastStartTime = startTime;
//            cout << "intervalTime: "<< intervalTime << endl;
            
/**********************************************************/
//! Generate the relative information  
/**********************************************************/
            /// Generate Lane Related Mass
            if (LANE_DETECTOR && LANE_ANALYZER)
            {
                /// First init the baseline, then get lane mass 
                if( pastTime < TIME_BASELINE ){
                    /// Get lane baseline
                    LaneDetector::GetLaneBaseline(sampleIdx, SAMPLING_TIME,
                                    muWindowSize, sigmaWindowSize,
                                    lateralOffset, LATSDBaselineVec,
                                    lateralOffsetDeque, LANEXDeque,
                                    TLCDeque, laneFeatures, intervalTime);
                    //idx_baseline = sampleIdx;
                }
                else {
                    //sampleIdx -= idx_baseline;
                    LaneDetector::GenerateLaneIndicators(sampleIdx, SAMPLING_TIME,
                                     muWindowSize, sigmaWindowSize,
                                     lateralOffset,
                                     lateralOffsetDeque,
                                     LANEXDeque, TLCDeque, 
                                     laneFeatures, intervalTime);
                    //! LATSD
                    char *text_LATSD = new char[30];
                    sprintf(text_LATSD, "L1. LATSD: %.4f", laneFeatures.LATSD);
                    cv::putText(laneMat, text_LATSD, cv::Point(0, 70), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_LATSD;
                    //! LATMEAN
                    char *text_LATMEAN = new char[30];
                    sprintf(text_LATMEAN, "L2. LATMEAN: %.4f", laneFeatures.LATMEAN);
                    cv::putText(laneMat, text_LATMEAN, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_LATMEAN;
                    //! LANEDEV
                    char *text_LANEDEV = new char[30];
                    sprintf(text_LANEDEV, "L3. LANEDEV: %.4f", laneFeatures.LANEDEV);
                    cv::putText(laneMat, text_LANEDEV, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_LANEDEV;
                    //! LANEX
                    char *text_LANEX = new char[30];
                    sprintf(text_LANEX, "L4. LANEX: %.4f", laneFeatures.LANEX);
                    cv::putText(laneMat, text_LANEX, cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_LANEX;
                    //! TLC
                    char *text_TLC = new char[30];
                    sprintf(text_TLC, "L5. TLC: %.4f", laneFeatures.TLC);
                    cv::putText(laneMat, text_TLC, cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_TLC;
                    //! TLC_2s
                    char *text_TLC_2s = new char[30];
                    sprintf(text_TLC_2s, "L6. TLC_2s: %d", laneFeatures.TLC_2s);
                    cv::putText(laneMat, text_TLC_2s, cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_TLC_2s;
                    
                    char *text_TLCF_2s = new char[50];
                    sprintf(text_TLCF_2s, "L7. Fraction_TLC_2s: %f", laneFeatures.TLCF_2s);
                    cv::putText(laneMat, text_TLCF_2s, cv::Point(0, 130), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_TLCF_2s;
                    //! TLC_halfs
                    char *text_TLC_halfs = new char[30];
                    sprintf(text_TLC_halfs, "L8. TLC_halfs: %d", laneFeatures.TLC_halfs);
                    cv::putText(laneMat, text_TLC_halfs, cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_TLC_halfs;
                    
                    char *text_TLCF_halfs = new char[50];
                    sprintf(text_TLCF_halfs, "L9. Fraction_TLC_halfs: %f", laneFeatures.TLCF_halfs);
                    cv::putText(laneMat, text_TLCF_halfs, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_TLCF_halfs;
                    //! TLC_min
                    char *text_TLC_min = new char[30];
                    sprintf(text_TLC_min, "L10. TLC_min: %.4f", laneFeatures.TLC_min);
                    cv::putText(laneMat, text_TLC_min, cv::Point(0, 160), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                    delete text_TLC_min;
                    
                }//end if
            }//end Generate Lane Indicators
            
            if(IMAGE_RECORD){
                char *text = new char[100];
                sprintf(text, LANE_RECORD_NAME, idx);
                cv::imwrite(text, laneMat);
                delete text;
            }
            
            cv::imshow("Lane Marking", laneMat);
            cv::moveWindow("Lane Marking", 790, 0);
            cv::waitKey(0);
            
            
            sampleIdx++;//update the sampling index
            idx++;
            if(idx == FRAME_STOP1)
                idx = FRAME_RESTART1;
            if(idx == FRAME_STOP2)
                idx = FRAME_RESTART2;
            
            //! Record the features
            if (DATA_RECORD) {
                //!Lane Features
                RecordLaneFeatures(laneFeatureFile, laneFeatures, pastTime);
            }//end if

            /**********************************************************/
            //! Send the datas as string to fusion center
            /**********************************************************/
            if(LANE_DETECTOR & SEND_DATA) {
                char *str = new char[1024];
                memset(str, 0, 1024);
                CodeMsg(laneFeatures, str);
                
                strcpy(laneMsg.text, str);//!!! overflow 
                laneMsg.type = 1;
                //printf("Data sends: %s\n", laneMsg.text);
                
                //! 0 will cause a block/ IPC_NOWAIT will close the app.
                if(msgsnd(mq_id, &laneMsg, sizeof(laneMsg), 0) == -1)
                {  
                    printf("LaneDetectSim: msgsnd failed!\n");
                    perror("error");
                    _exit(1);  
                }  
                delete str;
            }
            
            /**********************************************************/
            //! Adjust the interval time within fixed frequency
            /**********************************************************/
			double execFreq;
            do {
				cv::waitKey(1);
                execTime = ((double)cv::getTickCount() - startTime)/cv::getTickFrequency();
				execFreq = 1.0 / execTime;
            }
            while ( execFreq > SAMPLING_FREQ );
            
        }//end while loop 

        laneFeatureFile.close();
        cv::destroyAllWindows();
        
        return 0;
    }
}//FusedCarSurveillanceSim

#endif //__cplusplus

using LaneDetectorSim::Process;
int main(int argc, const char * argv[])
{
    return Process(argc, argv);
}
