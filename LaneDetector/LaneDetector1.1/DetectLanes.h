//
//  detectlane.h
//  LaneDetector1.0
//
//  Created by Xuanpeng Li on 03/13.
//  Copyright (c) 2012 ESIEE-AMIENS. All rights reserved.
//

#ifndef LaneDetector_DetectLanes_h
#define LaneDetector_DetectLanes_h

#include <algorithm>
#include <iostream>
#include <list>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "ExtractFeatures.h"
#include "FittingCurve.h"
#include "LaneDetectorTools.h"

namespace LaneDetector{	
    /// Line structure with start and end points
	typedef struct _Lane {
		///start point
		cv::Point2d startPoint;
		///end point
		cv::Point2d endPoint;
	} Lane;
    
	///Structure to hold lane detector settings
	typedef struct _LaneDetectorConf
	{
        //Whether execute the IPM
        int isIPM;  //0: false, 1: ture
        //! Relative with Configuration of Camera
        double m;           //number of rows in the full image.
        double n;           //number of columns in the full image.
        double h;           //the height above the ground (meter).
        double alphaTot;    //the total half viewing angle of the camera.
        double alpha_u;
        double alpha_v;
        double theta0;      //the camera tilted angle (pitch angle)
        double rHorizon;
        
        
        double ipmX_min;    //meter in IPM x coordinate
        double ipmX_max;    //meter in IPM x coordinate
        double ipmY_max;    //meter in IPM y coordinate
        double ipmY_min;    //meter in IPM y coordinate
        double ipmStep;     //pixel per meter in y coordinate
        double mIPM;        //pixel in IPM y coordinate
        
		///The method to use for IPM interpolation
		int ipmInterpolation;
        
		///width of line we are detecting
		double lineWidth;
		///height of line we are detecting
		double lineHeight;
        
		///kernel size to use for filtering
		unsigned char kernelWidth;
		unsigned char kernelHeight;
		///lower quantile to use for thresholding the filtered image
		double lowerQuantile;
		///whether to return local maxima or just the maximum
		bool localMaxima;
		///the type of grouping to use: 0 for HV lines and 1 for Hough Transform
		unsigned char groupingType;
        ///the type of feature extracion: 0 for SOBEL_1, 1 for SOBEL_2, 2 for CANNY and 3 for LAPLACE 
        unsigned char filterType;
		///whether to binarize the thresholded image or use the
		///raw filtered image
		bool binarize;
		//unsigned char topClip;
		///threshold for line scores to declare as line
		double detectionThreshold;
		///whtehter to smooth the line scores detected or not
		bool smoothScores;
		///rMin, rMax and rhoStep for Hough Transform (pixels)
		double rhoMin, rhoMax, rhoStep;
		///thetaMin, thetaMax, thetaStep for Hough Transform (radians)
		double thetaMin, thetaMax, thetaStep;
		///portion of image height to add to y-coordinate of vanishing
		///point when computing the IPM image
		double ipmVpPortion;
		///get end points or not
		bool getEndPoints;
		///group nearby lines
		bool group;
		///threshold for grouping nearby lines
		double groupThreshold;
		///use RANSAC or not
		bool ransac;
		///RANSAC Line parameters
		int ransacLineNumSamples;
		int ransacLineNumIterations;
		int ransacLineNumGoodFit;
		double ransacLineThreshold;
		double ransacLineScoreThreshold;
		bool ransacLineBinarize;
		///half width to use for ransac window
		int ransacLineWindow;
		///RANSAC Spline parameters
		int ransacSplineNumSamples;
		int ransacSplineNumIterations;
		int ransacSplineNumGoodFit;
		double ransacSplineThreshold;
		double ransacSplineScoreThreshold;
		bool ransacSplineBinarize;
		int ransacSplineWindow;
		///degree of spline to use
		int ransacSplineDegree;
		///use a spline or straight line
		bool ransacSpline;
		bool ransacLine;
        
		///step used to pixelize spline in ransac
		double ransacSplineStep;
        
		///Overlap threshold to use for grouping of bounding boxes
		double overlapThreshold;
        
		///Angle threshold used for localization (cosine, 1: most restrictive,
		/// 0: most liberal)
		double localizeAngleThreshold;
		///Number of pixels to go in normal direction for localization
		int localizeNumLinePixels;
        
	} LaneDetectorConf;
    
    void IPMPreprocess(const cv::Mat &ipmMat, const LaneDetectorConf &laneDetectorConf, cv::Mat &thMat);
    
    void IPMDetectLanes(const cv::Mat &ipmMat, 
                        const LaneDetectorConf &laneDetectorConf, 
                        std::vector<Lane> &leftIPMLanes, std::vector<Lane> &rightIPMLanes,
                        cv::Mat &leftCoefs, cv::Mat &rightCoefs,
                        std::vector<cv::Point2d> &leftSampledPoints, std::vector<cv::Point2d> &rightSampledPoints, double &laneWidth);
    
	/*
     * This function tracks lanes in the input image. 
     * The returned lines are in a vector of Line objects, 
     * having start and end point in input image frame.
	 * \param image the input image
	 * \param the original image
	 * \param camera information
	 * \param lane detect configuration  
	 */
	void DetectLanes(const cv::Mat &laneMat, 
                     const LaneDetectorConf &laneDetectorConf, 
                     const int &offsetY,
                     std::vector<cv::Vec2f> &hfLanes,
                     std::vector<cv::Vec2f> &postHfLanes,
                     const int &laneKalmanIdx,
                     const double &isChangeLane);
    
    
		
    enum FilterType {
        SOBEL_FILTER_1  = 0,
        SOBEL_FILTER_2  = 1,
        SOBEL_FILTER_3  = 2,
        CANNY_FILTER    = 3,
        LAPLACE_FILTER  = 4,
        HOG_FILTER      = 5,
        DOG_FILTER      = 6
    };
    /*
	 * This function extracts the edge from various filters
	 */
	void FilterLanes(cv::Mat &laneMat, const LaneDetectorConf &laneDetectorConf);	
    
    
    #define GROUPING_TYPE_HV_LINES      0
    #define GROUPING_TYPE_HOUGH_LINES   1
	/*
     * This function extracts lines from the passed infiltered 
     *  and thresholded image
	 * \param image the input thresholded filtered image
	 * \param lines a vector of lines
	 * \param lineConf the conf structure
	 */
	void GetHfLanes(cv::Mat &laneMat, 
                    const LaneDetectorConf &laneDetectorConf, 
                    std::vector<cv::Vec2f> &hfLanes,
                    const cv::Mat &lLaneMask,
                    const cv::Mat &rLaneMask,
                    const int &isROI, const int &isChangeLane);
    /*
     * This function define the rule of sort in hough params.
     */
    bool sort_smaller(const cv::Vec2f &lane1, const cv::Vec2f &lane2);
       
    /*
     * This function gets the lateral offset of ego-vehicle.
     */
    void GetLateralOffset(const cv::Mat &laneMat, 
                          const double &leftPoint, 
                          const double &rightPoint,
                          double &lateralOffset);
    
	/*
	 * This function does initialization of the struct of laneDetectorConf
	 * \param laneDetectorConf the structure to fill
	 */
	void InitlaneDetectorConf(LaneDetectorConf &laneDetectorConf);
    
    /*
     * This function enhance the adaptive contrast of images.
     */
    void EnhanceContrast_LCE(const cv::Mat &inMat, cv::Mat &outMat, const int &threshold = 1, const int &amount = 1);
    
    
    void hfLanetoLane(const cv::Mat &laneMat, const std::vector<cv::Vec2f> &hfLanes, std::vector<Lane> &lanes);
    
    void GetMarkerPoints(const cv::Mat &laneMat, const std::vector<cv::Vec2f> &hfLanes, cv::Point2d &vp, cv::Point2d &corner_l, cv::Point2d &corner_r, const int &offsetY);
    
    void DrawPreROI(cv::Mat &laneMat, 
                    const std::vector<cv::Vec2f> &postHfLanes, 
                    const int &laneKalmanIdx,
                    const int &isChangeLane);
    
    void DrawMarker(cv::Mat &laneMat, 
                    const std::vector<cv::Vec2f> &hfLanes, 
                    const double &lateralOffset);
    
    void MeasureLaneWidth(const std::vector<cv::Point2d> &leftSampledPoints, const std::vector<cv::Point2d> &rightSampledPoints, const LaneDetectorConf &laneDetectorConf, double &laneWidth);
}

#endif
