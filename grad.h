#pragma once
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

void CalcPatchDerivatives(const cv::Mat &patch, cv::Mat &patchDx, cv::Mat &patchDy);

void CalcPatchMergeDerivatives(const cv::Mat &patch, cv::Mat &patchDxy);


void ComputeHessian(
        const cv::Mat &dxPatch,
        const cv::Mat &dyPatch,
        double &A11,
        double &A12,
        double &A22);

void ComputeHessianFloat(
        const cv::Mat &dxPatch,
        const cv::Mat &dyPatch,
        float &A11,
        float &A12,
        float &A22);

void ComputeHessianFloatAOS(
        const cv::Mat &dxyPatch,
        float &A11,
        float &A12,
        float &A22);

bool ComputeSSD(
        const cv::Mat &prevPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        const int winSz,
        long &cost);

bool ComputeSSDSIMD(
        const cv::Mat &prevPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        const int winSz,
        float &cost);

bool ComputeResiduals(
        const cv::Mat &prevPatch,
        const cv::Mat &prevDxPatch,
        const cv::Mat &prevDyPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        float &b1,
        float &b2,
        int winSz);
        
bool ComputeResidualsSIMD(
        const cv::Mat &prevPatch,
        const cv::Mat &prevDxPatch,
        const cv::Mat &prevDyPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        float &b1,
        float &b2,
        int winSz);
        

void CopyToPatch(
        cv::Mat &prevPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        const int winSz);
