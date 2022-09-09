#include "grad.h"
#include <benchmark/benchmark.h>
#include <cstdlib>


static void DoSetup(cv::Mat patch, int type = 2) {
    for(int x = 0; x < patch.cols; x++) {
        for(int y = 0; y < patch.rows; y++) {
            if (type == 2) {
                patch.at<short>(y, x) =  rand() % 256;
            } else {
                patch.at<uchar>(y, x) =  rand() % 256;
            }
            
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

static void BM_ComputeHessianSIMDAOS(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat patchDxy(winSz, winSz, CV_16SC2);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);
        CalcPatchMergeDerivatives(patch, patchDxy);
        state.ResumeTiming();
        float A11, A12, A22;
        ComputeHessianFloatAOS(patchDxy, A11, A12, A22);
        benchmark::DoNotOptimize(A11);
        benchmark::DoNotOptimize(A22);
        benchmark::DoNotOptimize(A12);
    }
}

static void BM_ComputeSSD(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat nextImg(720, 720, CV_16SC1);
    DoSetup(nextImg);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);    
        cv::Point2f p;
        p.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        p.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        long cost;
        state.ResumeTiming();
        ComputeSSD(patch, nextImg, p, winSz, cost);
        benchmark::DoNotOptimize(cost);
    }
}

static void BM_ComputeSSDSIMD(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat nextImg(720, 720, CV_8UC1);
    DoSetup(nextImg, 1);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);    
        cv::Point2f p;
        p.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        p.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        float cost;
        state.ResumeTiming();
        ComputeSSDSIMD(patch, nextImg, p, winSz, cost);
        benchmark::DoNotOptimize(cost);
    }
}

static void BM_ComputeRandomCopy(benchmark::State& state) {
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat nextImg(720, 720, CV_8UC1);
    DoSetup(nextImg, 1);
    for(auto _ :state) {
        state.PauseTiming(); 
        DoSetup(patch);    
        cv::Point2f p;
        p.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        p.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        float cost;
        state.ResumeTiming();
        CopyToPatch(patch, nextImg, p, winSz);
        benchmark::DoNotOptimize(cost);
    }
}

static void BM_ComputeResiduals(benchmark::State& state){
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat patchDx(winSz, winSz, CV_16SC1);
    cv::Mat patchDy(winSz, winSz, CV_16SC1);
    cv::Mat nextImg(720, 720, CV_8UC1);
    DoSetup(nextImg, 1);
    for(auto _ : state) {
        state.PauseTiming(); 
        DoSetup(patch);
        CalcPatchDerivatives(patch, patchDx, patchDy);
        cv::Point2f p;
        p.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        p.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        float cost;
        state.ResumeTiming();
        float b1, b2;
        ComputeResiduals(patch, patchDx, patchDy, nextImg, p, b1, b2, winSz);
    }
}

static void BM_ComputeResidualsSIMD(benchmark::State& state){
    int winSz = 16;
    cv::Mat patch(winSz, winSz, CV_16SC1);
    cv::Mat patchDx(winSz, winSz, CV_16SC1);
    cv::Mat patchDy(winSz, winSz, CV_16SC1);
    cv::Mat nextImg(720, 720, CV_8UC1);
    DoSetup(nextImg, 1);
    for(auto _ : state) {
        state.PauseTiming(); 
        DoSetup(patch);
        CalcPatchDerivatives(patch, patchDx, patchDy);
        cv::Point2f p;
        p.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        p.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 720;
        float cost;
        state.ResumeTiming();
        float b1, b2;
        ComputeResidualsSIMD(patch, patchDx, patchDy, nextImg, p, b1, b2, winSz);
    }
}


BENCHMARK(BM_grad_benchmark);
BENCHMARK(BM_grad_benchmarkMerge);
BENCHMARK(BM_ComputeHessian);
BENCHMARK(BM_ComputeHessianSIMD);
BENCHMARK(BM_ComputeHessianSIMDAOS);
BENCHMARK(BM_ComputeSSD);
BENCHMARK(BM_ComputeSSDSIMD);
BENCHMARK(BM_ComputeRandomCopy);
BENCHMARK(BM_ComputeResiduals);
BENCHMARK(BM_ComputeResidualsSIMD);



BENCHMARK_MAIN();