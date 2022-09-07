#include "grad.h"
#include <benchmark/benchmark.h>
#include <cstdlib>


static void DoSetup(cv::Mat patch) {
    for(int x = 0; x < patch.cols; x++) {
        for(int y = 0; y < patch.rows; y++) {
            patch.at<short>(y, x) =  rand() % 256;
        }
    }
}

static void BM_grad_benchmark(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    
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
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat patchDxy(winSz, winSz, CV_16SC2);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);
        state.ResumeTiming();
        CalcPatchMergeDerivatives(patch, patchDxy);
    }
}


static void BM_ComputeHessian(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat patchDx(winSz, winSz, CV_16SC1);
    cv::Mat patchDy(winSz, winSz, CV_16SC1);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);
        CalcPatchDerivatives(patch, patchDx, patchDy);
        state.ResumeTiming();
        double A11, A12, A22;
        ComputeHessian(patchDx, patchDy, A11, A12, A22);
        benchmark::DoNotOptimize(A11);
        benchmark::DoNotOptimize(A22);
        benchmark::DoNotOptimize(A12);
    }
}

static void BM_ComputeHessianSIMD(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat patchDx(winSz, winSz, CV_16SC1);
    cv::Mat patchDy(winSz, winSz, CV_16SC1);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);
        CalcPatchDerivatives(patch, patchDx, patchDy);
        state.ResumeTiming();
        float A11, A12, A22;
        ComputeHessianFloat(patchDx, patchDy, A11, A12, A22);
        benchmark::DoNotOptimize(A11);
        benchmark::DoNotOptimize(A22);
        benchmark::DoNotOptimize(A12);
    }
}


BENCHMARK(BM_grad_benchmark);
BENCHMARK(BM_grad_benchmarkMerge);
BENCHMARK(BM_ComputeHessian);
BENCHMARK(BM_ComputeHessianSIMD);

BENCHMARK_MAIN();