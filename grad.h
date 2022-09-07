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