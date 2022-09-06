#pragma once
#include <opencv2/opencv.hpp>

void CalcPatchDerivatives(const cv::Mat &patch, cv::Mat &patchDx, cv::Mat &patchDy);

void CalcPatchMergeDerivatives(const cv::Mat &patch, cv::Mat &patchDxy);