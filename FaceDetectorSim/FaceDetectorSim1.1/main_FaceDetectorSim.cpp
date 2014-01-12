//
//  main.cpp
//  FaceDetectorSim1.1
//
//  Created by XUANPENG LI on 23/09/13.
//  Copyright (c) 2013 ESIEE-Amiens. All rights reserved.
//
#include "main_FaceDetectorSim.h"
#include "Process_FaceDetectorSim.h"

#ifdef __cplusplus
///Time
extern const double SAMPLING_TIME       = 60;
extern const double SAMPLING_FREQ       = 9.4;
extern const double TIME_BASELINE       = 300; //sec

/// Face feature analysis
extern const double TIME_PERCLOS_WINDOW = 60;
extern const double TIME_BLINK_WINDOW   = 60;
extern const double THRESHOLD_PERCLOS   = 0.8;
extern const double THRESHOLD_CLOSURE   = 0.7;
/// Size of Image
extern const double COEF                = 0.75;
extern const int    WIDTH               = 640;
extern const int    HEIGHT              = 480;
/// Multi-Image Show
extern const int    WIN_COLS            = 4;
extern const int    WIN_ROWS            = 3;

/// Capture images
extern const int    IMAGE_RECORD        = 0; // Record Face images
extern const int    EYES_RECORD         = 0; // Record eyes images

#ifdef __APPLE__
extern const char   FACE_IMG_NAME[] 	= "/Users/xuanpengli/Developer/Capture/face/face_%d.jpg";
extern const char   EYE_COLOR_NAME[]    = "/Users/xuanpengli/Developer/Capture/Eye_color/eye_color_%d_%d.jpg";
extern const char   EYE_BIN_NAME[]      = "/Users/xuanpengli/Developer/Capture/Eye_bin/eye_bin_%d_%d.jpg";
/// Record docs
extern const char   FILE_FACE_FEATURE[] = "/Users/xuanpengli/Developer/Record/sim_faceFeature.txt";
/// Data Source
extern const char   FACE_RAW_NAME[]     = "/Users/xuanpengli/Developer/Data/FaceRaw_10-07-2013_18h30m21s/face_%d.jpg";
extern const char   KEY_PATH[]          = "/Users/xuanpengli/Developer/key.txt";
#elif defined __linux
extern const char   FACE_IMG_NAME[] 	= "/home/lixp/Developer/Capture/face/face_%d.jpg";
extern const char   EYE_COLOR_NAME[]    = "/home/lixp/Developer/Capture/Eye_color/eye_color_%d_%d.jpg";
extern const char   EYE_BIN_NAME[]      = "/home/lixp/Developer/Capture/Eye_bin/eye_bin_%d_%d.jpg";
/// Record docs
extern const char   FILE_FACE_FEATURE[] = "/home/lixp/Developer/Record/sim_faceFeature.txt";
/// Data Source
extern const char   FACE_RAW_NAME[]     = "/home/lixp/Developer/Data/FaceRaw_10-07-2013_18h30m21s/face_%d.jpg";
extern const char   KEY_PATH[]          = "/home/lixp/Developer/key.txt";
#endif

/// Classifier Cascade 
extern const char   EYE_CASCADE_NAME[]  = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
extern const char   FACE_CASCADE_NAME[] = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

/// Frame 
extern const int    FRAME_START         = 2028;		//18:34:00 \2028
extern const int    FRAME_STOP1         = 15580;    //18:57:59
extern const int    FRAME_RESTART1      = 16712;    //19:00:00
extern const int    FRAME_STOP2         = 19539;    //19:04:59
extern const int    FRAME_RESTART2      = 21801;    //19:09:00
extern const int    FRAME_END           = 23511; 	//19:11:59


namespace FaceDetectorSim {
    int Process(int argc, const char* argv[])
	{
		/// Run application
		int FACE_DETECTOR   = atoi(argv[1]);
		int SEND_DATA       = atoi(argv[2]);
		int DATA_RECORD     = atoi(argv[3]); // Record the data
		int StartFrame		= atoi(argv[4]); // FRAME_START
		int EndFrame		= atoi(argv[5]); // FRAME_END
	
        int idx 			= StartFrame;  //index for image sequence
        int sampleIdx 		= 1;
        char faceImg[100];
        
        std::ofstream faceFeatureFile;
        if (DATA_RECORD){
            InitRecordData(faceFeatureFile, FILE_FACE_FEATURE, faceFeatureName, NUM_FACE);
        }
        
/**********************************************************/
// Parameters for Face Detector
/**********************************************************/
        cv::Mat faceMat;
        cv::Rect faceRoiRect;
        cv::CascadeClassifier faceCascade(FACE_CASCADE_NAME);
        cv::CascadeClassifier eyeCascade(EYE_CASCADE_NAME); 
        std::vector<double> eyesHeightBaselineVec;
        std::vector<double> eyesHeightVec;
        std::vector<cv::Point> eyesPointVec;
        std::deque<FaceDetector::InfoPERCLOS> PERCLOSDeque; 
        std::deque<FaceDetector::InfoBLINK> BLINKDeque;
        FaceDetector::FaceFeature faceFeatures;
        int faceKalmanIndex  = 0;
        
        //! Initialize Face Kalman Filter
        int trackKalman = 0;
        cv::KalmanFilter faceKalmanFilter(4,4,0);
        cv::Mat faceKalmanMeasureMat(4, 1, CV_32F, cv::Scalar::all(0));
        if (FACE_DETECTOR) {
            FaceDetector::InitFaceKalmanFilter(faceKalmanFilter, faceKalmanMeasureMat);       
        } 
        
        cv::Mat hist_camshift;
        cv::MatND faceHist;
        std::vector<FaceDetector::PARTICLE_FACE> particles;
        
/**********************************************************/        
//! Inter-process communication
/**********************************************************/
        key_t ipckey;
        struct { 
            long type; 
            char text[1024]; 
        } faceMsg;
        int mq_id;
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
                printf("FaceDetector creates a new MQ %d\n", mq_id);
            }
            else {
                //MQ does exit
                mq_id = msgget(ipckey, IPC_EXCL | 0666);
                printf("FaceDetector uses an existed MQ %d\n", mq_id);
            }
            //printf("Lane identifier is %d\n", mq_id);
            if(mq_id == -1) {  
                perror("error");
                _exit(1);  
            }
        }

        
/***************************************************/
//! Entrance of Face Detector
/***************************************************/
        if (faceCascade.empty() | eyeCascade.empty())
            abort ();
        else
        {
            double initTime         = (double)cv::getTickCount();
            double intervalTime     = 0;
            double execTime         = 0;  // Execute Time for Each Frame
            double pastTime         = 0;
            double lastStartTime    = (double)cv::getTickCount();
            while (idx <= EndFrame)
            {
                //printf("\nProcess in %d frames: \n", idx);
                double startTime = (double)cv::getTickCount();
                
                /**********************************************************/
                //!  Face detect and tracking 
                /**********************************************************/  
                if (FACE_DETECTOR)
                {
                    sprintf(faceImg, FACE_RAW_NAME, idx);
                    faceMat = cv::imread(faceImg);
                    
                    ProcessFaceImage(faceMat, faceCascade, eyeCascade, faceRoiRect, eyesHeightVec, eyesPointVec, startTime, idx, trackKalman, faceKalmanFilter, faceKalmanMeasureMat, faceKalmanIndex, execTime, hist_camshift, faceHist, particles);
                }
                /**********************************************************/  
                //! Calculate the running time for every sampling
                /**********************************************************/  
                //! past time
                pastTime = ((double)cv::getTickCount() - initTime)/cv::getTickFrequency();
                //printf("@Face Sampling passes %f sec\n", pastTime);
                char *text_pastTime = new char[30];
                sprintf(text_pastTime, "Time: %.2f sec", pastTime);
                cv::putText(faceMat, text_pastTime, cv::Point(0, faceMat.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,255,0));
                delete text_pastTime;
                
                intervalTime = (startTime - lastStartTime)/ cv::getTickFrequency();//get the time between two continuous frames
                lastStartTime = startTime;
                // cout << "intervalTime: "<< intervalTime << endl;
                
                /**********************************************************/
                //! Generate the relative information  
                /**********************************************************/
                /// Don't calculate the first sampling
                /// Calculate the mass after BASELINE_FRAME
                if (FACE_DETECTOR) {
                    if ( pastTime < TIME_BASELINE) {
                        /// Get face baseline
                        /// It cannot work at this version.
                        FaceDetector::GetFaceBaseline(sampleIdx,
                                        eyesHeightVec, 
                                        eyesHeightBaselineVec,
                                        faceFeatures);
                    }
                    else {
                        /// Generate face mass after baseline acquired 
                        FaceDetector::GenerateFaceIndicators(sampleIdx,
                                               eyesHeightVec, 
                                               faceFeatures, 
                                               PERCLOSDeque, 
                                               BLINKDeque, 
                                               intervalTime);
                        
                        //! eyesHeight
                        char *text_eyesHeight = new char[30];
                        sprintf(text_eyesHeight, "Eyes Height: %.2f", faceFeatures.Height);
                        cv::putText(faceMat, text_eyesHeight, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_eyesHeight;
                        //! PERCLOS
                        char *text_PERCLOS = new char[30];
                        sprintf(text_PERCLOS, "F1. PERCLOS: %.4f", faceFeatures.PERCLOS);
                        cv::putText(faceMat, text_PERCLOS, cv::Point(0, 70), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_PERCLOS;
                        //! MICROSLEEP
                        char *text_MICROSLEEP = new char[30];
                        sprintf(text_MICROSLEEP, "F2. MICROSLEEP: %.4f", faceFeatures.MICROSLEEP);
                        cv::putText(faceMat, text_MICROSLEEP, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_MICROSLEEP;
                        //! BLINK
                        char *text_BLINK = new char[30];
                        sprintf(text_BLINK, "F3. BLINK: %.4f", faceFeatures.BLINK);
                        cv::putText(faceMat, text_BLINK, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0,255,0));
                        delete text_BLINK;
                        
                        //! Eye states
                        char *text_eye = new char[30];
                        if (BLINKDeque.back().eyeBLINK == 0) {
                            sprintf(text_eye, "%s", "Eyes Open");
                            cv::putText(faceMat, text_eye, cv::Point(180, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,0,255));
                        } else {
                            sprintf(text_eye, "%s", "Eyes Closure");
                            cv::putText(faceMat, text_eye, cv::Point(180, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255,0,0));
                        }
                        delete text_eye;
                        
                    }//end Generate Face Indicators
                    
                   
                    
                    cv::imshow("Face", faceMat);
                    cv::moveWindow("Face", 790, 410);
                    cv::waitKey(1);
                    
                    if(IMAGE_RECORD){
                        char *faceImg = new char[100]; 
                        sprintf(faceImg, FACE_IMG_NAME, idx);
                        cv::imwrite(faceImg, faceMat);
                        delete faceImg;
                    }
                    
                }//end FACE_DETECTOR
                
                
                
                //! Update the sampling index
                sampleIdx++;
                idx++;
                if(idx == FRAME_STOP1)
                    idx = FRAME_RESTART1;
                if(idx == FRAME_STOP2)
                    idx = FRAME_RESTART2;
                
                //! Record the features
                if (DATA_RECORD) {
                    //!Face Features
                    RecordFaceFeatures(faceFeatureFile, faceFeatures, pastTime);
                }//end DATA_FUSION
                
                /**********************************************************/
                //! Send the datas as string to fusion center
                /**********************************************************/
                if(FACE_DETECTOR & SEND_DATA) {
                    char *str = new char[1024];
                    memset(str, 0, 1024);
                    CodeMsg(faceFeatures, str);
    
                    strcpy(faceMsg.text, str);
                    faceMsg.type = 1;
                    //! 0 will cause a block/ IPC_NOWAIT will close the app.
                    if(msgsnd(mq_id, &faceMsg, sizeof(faceMsg), 0) == -1)
                    {  
                        printf("FaceDetectSim: msgsnd failed!\n");
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
                //printf("Process For Face Smapling in %f sec, about %f Hz.\n", execTime, 1/execTime);
                
            }//end while
        }//end if
            
        faceFeatureFile.close();
        cv::destroyAllWindows();
        
        return 0;
    }//end Process
}//FaceDetectorSim

#endif //__cplusplus

using FaceDetectorSim:: Process;
int main(int argc, const char * argv[])
{
    return Process(argc, argv);
}
