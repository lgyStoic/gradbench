#include "grad.h"
#include <benchmark/benchmark.h>
#include <cstdlib>


static void DoSetup(cv::Mat patch) {
    for(int x = 0; x < patch.cols; x++) {
        for(int y = 0; y < patch.rows; y++) {
            patch.at<uchar>(y, x) =  rand() % 256;
        }
    }
}

static void BM_grad_benchmark(benchmark::State& state) {
    int winSz = 15;
    cv::Mat patch(winSz, winSz, CV_8UC1);
    
    cv::Mat patchDx(winSz, winSz, CV_16SC1);
    cv::Mat patchDy(winSz, winSz, CV_16SC1);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);
        state.ResumeTiming(); 
        CalcPatchDerivatives(patch, patchDx, patchDy);
    }
}

static void BM_grad_benchmarkMerge(benchmark::State& state) {
    int winSz = 15;
    cv::Mat patch(winSz, winSz, CV_8UC1);
    
    
    cv::Mat patchDxy(winSz, winSz, CV_16SC2);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);
        state.ResumeTiming();
        CalcPatchMergeDerivatives(patch, patchDxy);
    }
}

BENCHMARK(BM_grad_benchmark);
BENCHMARK(BM_grad_benchmarkMerge);

BENCHMARK_MAIN();