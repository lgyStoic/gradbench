#include "grad.h"
using namespace cv;
using deriv_type = short;
void CalcPatchDerivatives(const cv::Mat &patch, cv::Mat &patchDx, cv::Mat &patchDy) {
    const int rows = patch.rows;
    const int cols = patch.cols;

    const int delta = (int)alignSize((cols + 2), 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + 1, 16), *trow1 = alignPtr(trow0 + delta, 16);

    for(int y = 0; y < rows; y++ )
    {
        // reflect rows in border
        const auto srow0 = patch.ptr<deriv_type>(y > 0 ? y-1 : 1);
        const auto srow1 = patch.ptr<deriv_type>(y);
        const auto srow2 = patch.ptr<deriv_type>(y < rows-1 ? y+1 : rows-2);
        deriv_type* dxrow = (deriv_type *)patchDx.ptr<deriv_type>(y);
        deriv_type* dyrow = (deriv_type *)patchDy.ptr<deriv_type>(y);

        // do vertical convolution
        
        for(int x = 0 ; x < cols; x++ )
        {
            int t0 = (srow0[x] + srow2[x]) + srow1[x]*2;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        trow0[-1] = trow0[1]; trow0[cols] = trow0[cols - 2];
        trow1[-1] = trow1[1]; trow1[cols] = trow1[cols - 2];

        // do horizontal convolution, interleave the results and store them to dst
        for(int x = 0; x < cols; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+1] - trow0[x-1]);
            deriv_type t1 = (deriv_type)((trow1[x+1] + trow1[x-1]) + trow1[x]*2);
            dxrow[x] = t0;
            dyrow[x] = t1;
        }
    }

}

void CalcPatchMergeDerivatives(const cv::Mat &patch, cv::Mat &patchDxy) {
    const int rows = patch.rows;
    const int cols = patch.cols;

    const int delta = (int)alignSize((cols + 2), 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + 1, 16), *trow1 = alignPtr(trow0 + delta, 16);

    for(int y = 0; y < rows; y++ )
    {
        // reflect rows in border
        const auto srow0 = patch.ptr<deriv_type>(y > 0 ? y-1 : 1);
        const auto srow1 = patch.ptr<deriv_type>(y);
        const auto srow2 = patch.ptr<deriv_type>(y < rows-1 ? y+1 : rows-2);
        deriv_type* dxyrow = (deriv_type *)patchDxy.ptr<deriv_type>(y);

        // do vertical convolution
        
        for(int x = 0 ; x < cols; x++ )
        {
            int t0 = (srow0[x] + srow2[x]) + srow1[x]*2;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        trow0[-1] = trow0[1]; trow0[cols] = trow0[cols - 2];
        trow1[-1] = trow1[1]; trow1[cols] = trow1[cols - 2];

        // do horizontal convolution, interleave the results and store them to dst
        for(int x = 0; x < cols; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+1] - trow0[x-1]);
            deriv_type t1 = (deriv_type)((trow1[x+1] + trow1[x-1]) + trow1[x]*2);
            dxyrow[x] = t0;
            dxyrow[x + 1] = t1;
        }
    }
}
