#include "grad.h"
using deriv_type = short;
const int W_BITS = 14;
#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

void ComputeHessian(
        const cv::Mat &dxPatch,
        const cv::Mat &dyPatch,
        double &A11,
        double &A12,
        double &A22) {
    A11 = 0;
    A12 = 0;
    A22 = 0;
    for( int y = 0; y < dxPatch.rows; y++ )
    {
        auto* dxPtr = dxPatch.ptr<deriv_type>(y, 0);
        auto* dyPtr = dyPatch.ptr<deriv_type>(y, 0);
        int x = 0;
        for(; x < dyPatch.cols; x++)
        {
            float ixval = dxPtr[x];
            float iyval = dyPtr[x];
            A11 += (double)ixval * ixval;
            A12 += (double)ixval * iyval;
            A22 += (double)iyval * iyval;
        }
    }
}


void ComputeHessianFloat(
        const cv::Mat &dxPatch,
        const cv::Mat &dyPatch,
        float &A11,
        float &A12,
        float &A22) {
    A11 = 0;
    A12 = 0;
    A22 = 0;
    float32x4_t a11_f4 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t a12_f4 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t a22_f4 = {0.0, 0.0, 0.0, 0.0};
    for( int y = 0; y < dxPatch.rows; y++ )
    {
        auto* dxPtr = dxPatch.ptr<deriv_type>(y, 0);
        auto* dyPtr = dyPatch.ptr<deriv_type>(y, 0);
        int x = 0;
        
        for(; x < dxPatch.cols; x+=8) {
            float16x8_t dx_s8 = vcvtq_f16_s16(vld1q_s16(dxPtr + x));
            float16x8_t dy_s8 = vcvtq_f16_s16(vld1q_s16(dyPtr + x));
            float32x4_t l_dxf4 = vcvt_f32_f16((vget_low_f16(dx_s8)));
            float32x4_t l_dyf4 = vcvt_f32_f16((vget_low_f16(dy_s8)));
            float32x4_t h_dxf4 = vcvt_f32_f16((vget_high_f16(dx_s8)));
            float32x4_t h_dyf4 = vcvt_f32_f16((vget_high_f16(dy_s8)));

            float32x4_t lxxf4 = l_dxf4 * l_dxf4;
            float32x4_t lyyf4 = l_dyf4 * l_dyf4;
            float32x4_t lxyf4 = l_dyf4 * l_dxf4;

            float32x4_t hxxf4 = h_dxf4 * h_dxf4;
            float32x4_t hyyf4 = h_dyf4 * h_dyf4;
            float32x4_t hxyf4 = h_dxf4 * h_dyf4;

            a11_f4 = vaddq_f32(a11_f4, hxxf4);
            a12_f4 = vaddq_f32(a12_f4, hxyf4);
            a22_f4 = vaddq_f32(a22_f4, hyyf4);
            a11_f4 = vaddq_f32(a11_f4, lxxf4);
            a12_f4 = vaddq_f32(a12_f4, lxyf4);
            a22_f4 = vaddq_f32(a22_f4, lyyf4);
        }    
    }
    float32x2_t a11_f2 = vadd_f32(vget_high_f32(a11_f4), vget_low_f32(a11_f4));
    float32x2_t a12_f2 = vadd_f32(vget_high_f32(a12_f4), vget_low_f32(a12_f4));
    float32x2_t a22_f2 = vadd_f32(vget_high_f32(a22_f4), vget_low_f32(a22_f4));
    A11 = vget_lane_f32(vpadd_f32(a11_f2, a11_f2), 0);
    A12 = vget_lane_f32(vpadd_f32(a12_f2, a12_f2), 0);
    A22 = vget_lane_f32(vpadd_f32(a22_f2, a22_f2), 0);
}

void ComputeHessianFloatAOS(
        const cv::Mat &dxyPatch,
        float &A11,
        float &A12,
        float &A22) {
    A11 = 0;
    A12 = 0;
    A22 = 0;
    float32x4_t a11_f4 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t a12_f4 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t a22_f4 = {0.0, 0.0, 0.0, 0.0};
    for( int y = 0; y < dxyPatch.rows; y++ )
    {
        auto* dxyPtr = dxyPatch.ptr<deriv_type>(y, 0);
        int x = 0;
        
        for(; x < dxyPatch.cols; x+=8) {
            int16x8x2_t dxy_s8 = vld2q_s16(dxyPtr + x * 2);
            float32x4_t l_dxf4 = vcvt_f32_f16(vcvt_f16_s16(vget_low_s16(dxy_s8.val[0])));
            float32x4_t l_dyf4 = vcvt_f32_f16(vcvt_f16_s16(vget_low_s16(dxy_s8.val[1])));
            float32x4_t h_dxf4 = vcvt_f32_f16(vcvt_f16_s16(vget_high_s16(dxy_s8.val[0])));
            float32x4_t h_dyf4 = vcvt_f32_f16(vcvt_f16_s16(vget_high_s16(dxy_s8.val[1])));

            float32x4_t lxxf4 = l_dxf4 * l_dxf4;
            float32x4_t lyyf4 = l_dyf4 * l_dyf4;
            float32x4_t lxyf4 = l_dyf4 * l_dxf4;

            float32x4_t hxxf4 = h_dxf4 * h_dxf4;
            float32x4_t hyyf4 = h_dyf4 * h_dyf4;
            float32x4_t hxyf4 = h_dxf4 * h_dyf4;

            a11_f4 = vaddq_f32(a11_f4, hxxf4);
            a12_f4 = vaddq_f32(a12_f4, hxyf4);
            a22_f4 = vaddq_f32(a22_f4, hyyf4);
            a11_f4 = vaddq_f32(a11_f4, lxxf4);
            a12_f4 = vaddq_f32(a12_f4, lxyf4);
            a22_f4 = vaddq_f32(a22_f4, lyyf4);
        }    
    }
    float32x2_t a11_f2 = vadd_f32(vget_high_f32(a11_f4), vget_low_f32(a11_f4));
    float32x2_t a12_f2 = vadd_f32(vget_high_f32(a12_f4), vget_low_f32(a12_f4));
    float32x2_t a22_f2 = vadd_f32(vget_high_f32(a22_f4), vget_low_f32(a22_f4));
    A11 = vget_lane_f32(vpadd_f32(a11_f2, a11_f2), 0);
    A12 = vget_lane_f32(vpadd_f32(a12_f2, a12_f2), 0);
    A22 = vget_lane_f32(vpadd_f32(a22_f2, a22_f2), 0);
}

bool ComputeSSD(
        const cv::Mat &prevPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        const int winSz,
        long &cost) {
    cost = 0;
    cv::Point2f corner = {nextPt.x - winSz / 2, nextPt.y - winSz / 2};
    int iCornerX = cvFloor(corner.x);
    int iCornerY = cvFloor(corner.y);
    if (iCornerX < 0 || iCornerX + winSz >= nextImg.cols ||
        iCornerY < 0 || iCornerY + winSz >= nextImg.rows) {
        return false;
    }

    float a = corner.x - iCornerX;
    float b = corner.y - iCornerY;
    int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
    int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
    int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    const int stepJ = static_cast<int>(nextImg.step1());

    for(int y = 0; y < winSz; y++)
    {
        const auto* patchPtr = prevPatch.ptr<deriv_type>(y, 0);
        const auto* next_ptr = nextImg.ptr<uchar>(y+iCornerY, iCornerX);
        int x = 0;
        for(; x < winSz; x++)
        {
            int diff = CV_DESCALE(next_ptr[x]*iw00 + next_ptr[x+1]*iw01 + next_ptr[x+stepJ]*iw10 +
                                  next_ptr[x+stepJ+1]*iw11, W_BITS-5) - patchPtr[x]; // real_diff * (2^5)
            cost += diff * diff;
        }
    }

    return true;
}

bool ComputeSSDSIMD(
        const cv::Mat &prevPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        const int winSz,
        long &cost) {
    cost = 0;
    cv::Point2f corner = {nextPt.x - winSz / 2, nextPt.y - winSz / 2};
    int iCornerX = cvFloor(corner.x);
    int iCornerY = cvFloor(corner.y);
    if (iCornerX < 0 || iCornerX + winSz >= nextImg.cols ||
        iCornerY < 0 || iCornerY + winSz >= nextImg.rows) {
        return false;
    }

    float a = corner.x - iCornerX;
    float b = corner.y - iCornerY;
    int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
    int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
    int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
    

    const int stepJ = static_cast<int>(nextImg.step1());

    for(int y = 0; y < winSz; y++)
    {
        const auto* patchPtr = prevPatch.ptr<deriv_type>(y, 0);
        const auto* next_ptr = nextImg.ptr<uchar>(y+iCornerY, iCornerX);
        int x = 0;
        for(; x < winSz; x++)
        {
            int diff = CV_DESCALE(next_ptr[x]*iw00 + next_ptr[x+1]*iw01 + next_ptr[x+stepJ]*iw10 +
                                  next_ptr[x+stepJ+1]*iw11, W_BITS-5) - patchPtr[x]; // real_diff * (2^5)
            cost += diff * diff;
        }
    }

    return true;
}