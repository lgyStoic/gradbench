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
        
        for(; x <= dxPatch.cols - 8; x+=8) {
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
        
        for(; x <= dxyPatch.cols - 8; x+=8) {
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
        float &cost) {
    cost = 0.0;
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
    int32x4_t iw00_4s32 = vdupq_n_s32(iw00);
    int32x4_t iw01_4s32 = vdupq_n_s32(iw01);
    int32x4_t iw10_4s32 = vdupq_n_s32(iw10);
    int32x4_t iw11_4s32 = vdupq_n_s32(iw11);

    float32x4_t costf8 = vdupq_n_f32(0.0);
    for(int y = 0; y < winSz; y++)
    {
        auto* patchPtr = prevPatch.ptr<deriv_type>(y, 0);
        auto* next_ptr = nextImg.ptr<uchar>(y+iCornerY, iCornerX);
        
        for(int x = 0; x <= winSz - 8; x += 8) {
            int16x8_t patch_8s16 =  vld1q_s16(patchPtr + x);

            int16x8_t next_00_16s8 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(next_ptr + x)));
            int16x8_t next_01_16s8 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(next_ptr + x + 1)));
            int16x8_t next_10_16s8 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(next_ptr + x + nextImg.cols)));
            int16x8_t next_11_16s8 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(next_ptr + x + nextImg.cols + 1)));
            
            int32x4_t x00h_4s32 = vmulq_s32(vmovl_s16(vget_high_s16(next_00_16s8)),  iw00_4s32);
            int32x4_t x01h_4s32 = vmulq_s32(vmovl_s16(vget_high_s16(next_01_16s8)),  iw01_4s32);
            
            int32x4_t x10h_4s32 = vmlaq_s32(x00h_4s32, vmovl_s16(vget_high_s16(next_10_16s8)),  iw10_4s32);
            int32x4_t x11h_4s32 = vmlaq_s32(x01h_4s32, vmovl_s16(vget_high_s16(next_11_16s8)),  iw11_4s32);
            
            
            int32x4_t hinterp_4s32 = vshrq_n_s32(vaddq_s32(x10h_4s32, x11h_4s32), 9);
            
            int32x4_t patch_h4s32 = vmovl_s16(vget_high_s16(patch_8s16));
            
            float32x4_t diff_h4f32 = vcvtq_f32_s32(vsubq_s32(hinterp_4s32, patch_h4s32));
            
            
            costf8 = vmlaq_f32(costf8, diff_h4f32, diff_h4f32);
            
            int32x4_t x00l_4s32 = vmulq_s32(vmovl_s16(vget_low_s16(next_00_16s8)),  iw00_4s32);
            int32x4_t x01l_4s32 = vmulq_s32(vmovl_s16(vget_low_s16(next_01_16s8)),  iw01_4s32);
            int32x4_t x10l_4s32 = vmlaq_s32(x00l_4s32, vmovl_s16(vget_low_s16(next_10_16s8)),  iw10_4s32);
            int32x4_t x11l_4s32 = vmlaq_s32(x01l_4s32, vmovl_s16(vget_low_s16(next_11_16s8)),  iw11_4s32);
            
            int32x4_t linterp_4s32 = vshrq_n_s32(vaddq_s32(x10l_4s32, x11l_4s32), 9);
            int32x4_t patch_l4s32 = vmovl_s16(vget_low_s16(patch_8s16));
            
            float32x4_t diff_l4f32 = vcvtq_f32_s32(vsubq_s32(linterp_4s32, patch_l4s32));
            costf8 = vmlaq_f32(costf8, diff_l4f32, diff_l4f32);
        }
    }
    float32x2_t f2 = vadd_f32(vget_high_f32(costf8), vget_low_f32(costf8));
    cost = vget_lane_f32(vpadd_f32(f2, f2), 0);
    return true;
}


bool ComputeResiduals(
        const cv::Mat &prevPatch,
        const cv::Mat &prevDxPatch,
        const cv::Mat &prevDyPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        float &b1,
        float &b2,
        int winSz) {
    b1 = 0;
    b2 = 0;

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
        const auto* dxPatchPtr = prevDxPatch.ptr<deriv_type>(y, 0);
        const auto* dyPatchPtr = prevDyPatch.ptr<deriv_type>(y, 0);
        const auto* next_ptr = nextImg.ptr<uchar>(y+iCornerY, iCornerX);
        int x = 0;

        for(; x < winSz; x++)
        {
            float diff = CV_DESCALE(next_ptr[x]*iw00 + next_ptr[x+1]*iw01 + next_ptr[x+stepJ]*iw10 +
                                     next_ptr[x+stepJ+1]*iw11, W_BITS-5) - patchPtr[x];    // real_diff * (2^5)
            b1 += diff * dxPatchPtr[x];     // real_b1 * (2^10)
            b2 += diff * dyPatchPtr[x];
        }
    }
    return true;
}


bool ComputeResidualsSIMD(
        const cv::Mat &prevPatch,
        const cv::Mat &prevDxPatch,
        const cv::Mat &prevDyPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        float &b1,
        float &b2,
        int winSz) {
    b1 = 0;
    b2 = 0;

    cv::Point2f corner = {nextPt.x - winSz / 2, nextPt.y - winSz / 2};
    int iCornerX = cvFloor(corner.x);
    int iCornerY = cvFloor(corner.y);
    if (iCornerX < 0 || iCornerX + winSz >= nextImg.cols ||
        iCornerY < 0 || iCornerY + winSz >= nextImg.rows) {
        return false;
    }

    float a = corner.x - iCornerX;
    float b = corner.y - iCornerY;
    float16_t fw00 = cvRound((1.f - a)*(1.f - b));
    float16_t fw01 = cvRound(a*(1.f - b));
    float16_t fw10 = cvRound((1.f - a)*b);
    float16_t fw11 = 1 - fw00 - fw01 - fw10;
    
    float16x8_t fw00_x8 = vdupq_n_f16(fw00);
    float16x8_t fw01_x8 = vdupq_n_f16(fw01);
    float16x8_t fw10_x8 = vdupq_n_f16(fw10);
    float16x8_t fw11_x8 = vdupq_n_f16(fw11);

    float32x4_t b1_f4 = vdupq_n_f32(0.0);
    float32x4_t b2_f4 = vdupq_n_f32(0.0);
    for(int y = 0; y < winSz; y++)
    {
        auto* patchPtr = prevPatch.ptr<deriv_type>(y, 0);
        auto* dxPatchPtr = prevDxPatch.ptr<deriv_type>(y, 0);
        auto* dyPatchPtr = prevDyPatch.ptr<deriv_type>(y, 0);
        auto* next_ptr = nextImg.ptr<uchar>(y+iCornerY, iCornerX);
        int x = 0;
        for(; x <= winSz - 16; x += 16)
        {
            float16x8_t patch_f8 =  vcvtq_f16_s16(vshrq_n_s16(vld1q_s16(patchPtr + x), 5));
            float16x8_t patch_a8_f8 = vcvtq_f16_s16(vshrq_n_s16(vld1q_s16(patchPtr + 8 + x), 5));

            uint8x16_t next_00x16 = vld1q_u8(next_ptr + x);
            uint8x16_t next_01x16 = vld1q_u8(next_ptr + x + 1);
            uint8x16_t next_10x16 = vld1q_u8(next_ptr + x + winSz);
            uint8x16_t next_11x16 = vld1q_u8(next_ptr + x + winSz + 1);

            float16x8_t patchDx_f8 = vcvtq_f16_s16(vshrq_n_s16(vld1q_s16(dxPatchPtr + x), 5));
            float16x8_t patchDy_f8 = vcvtq_f16_s16(vshrq_n_s16(vld1q_s16(dyPatchPtr + x), 5));

            float16x8_t next_00_lowf8 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(next_00x16)));
            float16x8_t next_01_lowf8 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(next_01x16)));
            float16x8_t next_10_lowf8 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(next_10x16)));
            float16x8_t next_11_lowf8 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(next_11x16)));

            float16x8_t interp0001_lowf8 = vaddq_f16(vmulq_f16(fw00_x8, next_00_lowf8), vmulq_f16(fw01_x8, next_01_lowf8));
            float16x8_t interp1011_lowf8 = vaddq_f16(vmulq_f16(fw10_x8, next_10_lowf8), vmulq_f16(fw11_x8, next_11_lowf8));
            float16x8_t interpNext_lowf8 = vaddq_f16(interp0001_lowf8, interp1011_lowf8);

            float16x8_t diff_lowf8 = vsubq_f16(interpNext_lowf8, patch_f8);
            
            float32x4_t dxbxdiff4 = vaddq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(patchDx_f8)), vcvt_f32_f16(vget_low_f16(diff_lowf8))), vmulq_f32(vcvt_f32_f16(vget_high_f16(patchDx_f8)), vcvt_f32_f16(vget_high_f16(diff_lowf8))));
            float32x4_t dybxdiff4 = vaddq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(patchDy_f8)), vcvt_f32_f16(vget_low_f16(diff_lowf8))), vmulq_f32(vcvt_f32_f16(vget_high_f16(patchDy_f8)), vcvt_f32_f16(vget_high_f16(diff_lowf8))));

            float16x8_t patchDxa8_f8 = vcvtq_f16_s16(vshrq_n_s16(vld1q_s16(dxPatchPtr + x + 8), 5));
            float16x8_t patchDya8_f8 = vcvtq_f16_s16(vshrq_n_s16(vld1q_s16(dyPatchPtr + x + 8), 5));

            float16x8_t next_00_highf8 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(next_00x16)));
            float16x8_t next_01_highf8 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(next_01x16)));
            float16x8_t next_10_highf8 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(next_10x16)));
            float16x8_t next_11_highf8 = vcvtq_f16_s16(vmovl_s8(vget_high_s8(next_11x16)));

            float16x8_t interp0001_highf8 = vaddq_f16(vmulq_f16(fw00_x8, next_00_highf8), vmulq_f16(fw01_x8, next_01_highf8));
            float16x8_t interp1011_highf8 = vaddq_f16(vmulq_f16(fw10_x8, next_10_highf8), vmulq_f16(fw11_x8, next_11_highf8));
            float16x8_t interpNext_highf8 = vaddq_f16(interp0001_highf8, interp1011_highf8);

            float16x8_t diff_highf8 = vsubq_f16(interpNext_highf8, patch_f8);
            b1_f4 = vaddq_f32(dxbxdiff4, vaddq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(patchDxa8_f8)), vcvt_f32_f16(vget_low_f16(diff_highf8))), vmulq_f32(vcvt_f32_f16(vget_high_f16(patchDxa8_f8)), vcvt_f32_f16(vget_high_f16(diff_highf8)))));
            b2_f4 = vaddq_f32(dybxdiff4, vaddq_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(patchDya8_f8)), vcvt_f32_f16(vget_low_f16(diff_highf8))), vmulq_f32(vcvt_f32_f16(vget_high_f16(patchDya8_f8)), vcvt_f32_f16(vget_high_f16(diff_highf8)))));
        }

        for(; x < winSz; x++){           
            float diff = next_ptr[x]*fw00 + next_ptr[x+1]*fw01 + next_ptr[x]*fw10 +
                                  next_ptr[x+1]*fw11 - (patchPtr[x]>>5);
            b1 += diff * dxPatchPtr[x];     // real_b1 * (2^10)
            b2 += diff * dyPatchPtr[x];
        }

        b1 += vget_lane_f32(vadd_f32(vget_high_f32(b1_f4), vget_low_f32(b1_f4)), 0);
        b2 += vget_lane_f32(vadd_f32(vget_high_f32(b2_f4), vget_low_f32(b2_f4)), 0);
    }
    return true;
}

void CopyToPatch(
        cv::Mat &prevPatch,
        const cv::Mat &nextImg,
        const cv::Point2f &nextPt,
        const int winSz) {
    cv::Point2f corner = {nextPt.x - winSz / 2, nextPt.y - winSz / 2};
    int iCornerX = cvFloor(corner.x);
    int iCornerY = cvFloor(corner.y);
    if (iCornerX < 0 || iCornerX + winSz >= nextImg.cols ||
        iCornerY < 0 || iCornerY + winSz >= nextImg.rows) {
        return;
    }
    cv::Mat roi(nextImg, cv::Rect(corner.x, corner.y, winSz, winSz));
    roi.convertTo(prevPatch, CV_16SC1);

}