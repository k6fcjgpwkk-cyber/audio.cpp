#pragma once
// Metal Shading Language source for the opt-in FlashHead color kernels.
// This header is plain text so host tests can inspect it. It does not include
// Metal or compile the shader.
namespace engine::models::soulx_flashhead {
inline constexpr const char kFlashHeadColorMetalShader[] = R"msl(
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#pragma STDC FP_CONTRACT OFF
// MSL 4.1 section 2.1 has no 64-bit floating-point type. Every numeric value
// here is 32-bit float. math_mode(safe) plus contract(off) keeps the Kahan and
// two-sum steps from being reassociated or fused (MSL spec 1.6.3).
// flashhead_color_stats matches color_fused_metal_f32_row: 256 lanes.

inline float mad(float x, float s, float b) { return fma(x, s, b); }
inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float power_f(float x, float exponent) {
    if (exponent == 3.0f) {
        float sq = x * x;
        return sq * x;
    }
    float safe = clampf(x, 1.0e-30f, 1.0e30f);
    return exp(log(safe) * exponent);
}
inline float step_gt(float x, float bound) {
    return mad(x, 1.0f, -bound) > 0.0f ? 1.0f : 0.0f;
}
inline float select1(float mask, float yes, float no) {
    float inv = mad(mask, -1.0f, 1.0f);
    return mask * yes + inv * no;
}
inline void mat3(constant float * m, float r, float g, float b, thread float & x, thread float & y, thread float & z) {
    x = fma(b, m[6], fma(g, m[3], fma(r, m[0], 0.0f)));
    y = fma(b, m[7], fma(g, m[4], fma(r, m[1], 0.0f)));
    z = fma(b, m[8], fma(g, m[5], fma(r, m[2], 0.0f)));
}
inline void rgb01_to_lab(constant float * k, float r, float g, float b, thread float & l, thread float & a, thread float & bb) {
    float rg = power_f(mad(r, k[21], k[22]), k[23]);
    float gg = power_f(mad(g, k[21], k[22]), k[23]);
    float bg = power_f(mad(b, k[21], k[22]), k[23]);
    float mr = step_gt(r, k[25]);
    float mg = step_gt(g, k[25]);
    float mb = step_gt(b, k[25]);
    float lr = select1(mr, rg, r * k[24]);
    float lg = select1(mg, gg, g * k[24]);
    float lb = select1(mb, bg, b * k[24]);
    float x, y, z;
    mat3(k, lr, lg, lb, x, y, z);
    x = clampf(x / k[18], k[29], 1.0f);
    y = clampf(y / k[19], k[29], 1.0f);
    z = clampf(z / k[20], k[29], 1.0f);
    float fx = select1(step_gt(x, k[26]), power_f(x, k[30]), mad(x, k[27], k[28]));
    float fy = select1(step_gt(y, k[26]), power_f(y, k[30]), mad(y, k[27], k[28]));
    float fz = select1(step_gt(z, k[26]), power_f(z, k[30]), mad(z, k[27], k[28]));
    l = mad(fy, k[31], k[32]);
    a = (fx - fy) * k[33];
    bb = (fy - fz) * k[34];
}
inline void lab_to_rgb01(constant float * k, float l, float a, float b, thread float & r, thread float & g, thread float & bb) {
    float fy = mad(l, k[35], k[36]);
    float fx = a * k[37] + fy;
    float fz = fy - b * k[38];
    float cube_x = (fx * fx) * fx;
    float cube_y = (fy * fy) * fy;
    float cube_z = (fz * fz) * fz;
    float x = select1(step_gt(cube_x, k[39]), cube_x, mad(fx, k[42], k[43]));
    float y = select1(step_gt(l, k[40]), cube_y, l * k[41]);
    float z = select1(step_gt(cube_z, k[39]), cube_z, mad(fz, k[42], k[43]));
    float rr, gg, bc;
    mat3(k + 9, x * k[18], y * k[19], z * k[20], rr, gg, bc);
    r = clampf(select1(step_gt(rr, k[47]), mad(power_f(rr, k[44]), k[45], k[46]), rr * k[48]), 0.0f, 1.0f);
    g = clampf(select1(step_gt(gg, k[47]), mad(power_f(gg, k[44]), k[45], k[46]), gg * k[48]), 0.0f, 1.0f);
    bb = clampf(select1(step_gt(bc, k[47]), mad(power_f(bc, k[44]), k[45], k[46]), bc * k[48]), 0.0f, 1.0f);
}
inline device const float * shifted(device const float * base, constant uint * u, uint lo, uint hi) {
    ulong off = ((ulong)u[hi] << 32) | (ulong)u[lo];
    return base + (off >> 2);
}
inline device float * shifted_rw(device float * base, constant uint * u, uint lo, uint hi) {
    ulong off = ((ulong)u[hi] << 32) | (ulong)u[lo];
    return base + (off >> 2);
}

kernel void flashhead_color_lab(
        device const float * src [[buffer(0)]],
        device float * dst [[buffer(1)]],
        constant uint * u [[buffer(2)]],
        constant float * k [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    uint pixels = u[5];
    uint frames = u[4];
    if (gid >= frames * pixels) return;
    uint f = gid / pixels;
    uint p = gid - f * pixels;
    uint t = u[3] + f;
    uint in_frames = u[2];
    device const float * video = shifted(src, u, 6, 7);
    device float * lab = shifted_rw(dst, u, 8, 9);
    float rgb[3];
    for (uint c = 0; c < 3; ++c)
        rgb[c] = mad(video[((c * in_frames + t) * pixels) + p], 0.5f, 0.5f);
    float l, a, b;
    rgb01_to_lab(k, rgb[0], rgb[1], rgb[2], l, a, b);
    lab[(f * 3u) * pixels + p] = l;
    lab[(f * 3u + 1u) * pixels + p] = a;
    lab[(f * 3u + 2u) * pixels + p] = b;
}

inline float kahan_add(thread float & sum, thread float & comp, float value) {
    float y = value - comp;
    float t = sum + y;
    float recovered = t - sum;
    comp = recovered - y;
    sum = t;
    return sum;
}

struct FloatPair { float sum; float error; };
inline FloatPair two_sum(float a, float b) {
    // Returns the exact sum of two floats. A struct avoids taking a thread
    // reference to threadgroup memory, which MSL rejects.
    float s = a + b;
    float b_virtual = s - a;
    float a_virtual = s - b_virtual;
    FloatPair out;
    out.sum = s;
    out.error = (a - a_virtual) + (b - b_virtual);
    return out;
}

kernel void flashhead_color_stats(
        device const float * lab_base [[buffer(0)]],
        device float * stats_base [[buffer(1)]],
        constant uint * u [[buffer(2)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tg [[threadgroup_position_in_grid]]) {
    uint pixels = u[0];
    uint frames = u[1];
    // The host dispatches exactly frames*3 groups of 256 threads, so every
    // thread in the group reaches the barriers below.
    uint f = tg / 3u;
    uint c = tg - f * 3u;
    threadgroup float hi[256];
    threadgroup float lo[256];
    threadgroup float mean_shared;
    device const float * lab = shifted(lab_base, u, 2, 3);
    device const float * row = lab + ((f * 3u + c) * pixels);
    float sum = 0.0f;
    float comp = 0.0f;
    for (uint p = tid; p < pixels; p += 256u)
        kahan_add(sum, comp, row[p]);
    hi[tid] = sum;
    lo[tid] = -comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            FloatPair merged = two_sum(hi[tid], hi[tid + stride]);
            float low = (lo[tid] + lo[tid + stride]) + merged.error;
            FloatPair renorm = two_sum(merged.sum, low);
            hi[tid] = renorm.sum;
            lo[tid] = renorm.error;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u)
        mean_shared = (hi[0] + lo[0]) / float(pixels);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = mean_shared;
    sum = 0.0f;
    comp = 0.0f;
    for (uint p = tid; p < pixels; p += 256u) {
        float delta = row[p] - mean;
        kahan_add(sum, comp, delta * delta);
    }
    hi[tid] = sum;
    lo[tid] = -comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            FloatPair merged = two_sum(hi[tid], hi[tid + stride]);
            float low = (lo[tid] + lo[tid + stride]) + merged.error;
            FloatPair renorm = two_sum(merged.sum, low);
            hi[tid] = renorm.sum;
            lo[tid] = renorm.error;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        device float * stats = shifted_rw(stats_base, u, 4, 5);
        uint n = frames * 3u;
        stats[f * 3u + c] = mean;
        float variance = (hi[0] + lo[0]) / float(pixels);
        if (variance < 0.0f)
            variance = 0.0f;
        stats[n + f * 3u + c] = sqrt(variance);
    }
}

kernel void flashhead_color_match(
        device const float * src [[buffer(0)]],
        device const float * lab_base [[buffer(1)]],
        device float * dst [[buffer(2)]],
        device float * motion_base [[buffer(3)]],
        constant uint * u [[buffer(4)]],
        constant float * k [[buffer(5)]],
        device const float * stats [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    uint pixels = u[5];
    uint frames = u[4];
    if (gid >= frames * pixels) return;
    uint f = gid / pixels;
    uint p = gid - f * pixels;
    uint t = u[3] + f;
    uint in_frames = u[2];
    device const float * video = shifted(src, u, 6, 7);
    device const float * lab = shifted(lab_base, u, 8, 9);
    // Output byte offset is u[14], u[15]. Motion byte offset is u[16], u[17].
    device float * out = shifted_rw(dst, u, 14, 15);
    float corrected[3];
    for (uint c = 0; c < 3; ++c) {
        float value = lab[((f * 3u + c) * pixels) + p];
        device const float * reduced = shifted(stats, u, 18, 19);
        uint n = frames * 3u;
        float mean = reduced[f * 3u + c];
        float dev = reduced[n + f * 3u + c];
        float safe = dev < k[57] ? 1.0f : dev;
        float ratio = k[54 + c] / safe;
        float centered = value - mean;
        float scaled = centered * ratio;
        corrected[c] = scaled + k[51 + c];
    }
    float rr, gg, bb;
    lab_to_rgb01(k, corrected[0], corrected[1], corrected[2], rr, gg, bb);
    float rgb01[3] = {rr, gg, bb};
    float inverse = k[50];
    float strength = k[49];
    uint motion_frames = u[12];
    device float * motion = shifted_rw(motion_base, u, 16, 17);
    for (uint c = 0; c < 3; ++c) {
        float source = mad(video[((c * in_frames + t) * pixels) + p], 0.5f, 0.5f);
        float blended = source * inverse + rgb01[c] * strength;
        float sample = mad(blended, 2.0f, -1.0f);
        out[((c * frames + f) * pixels) + p] = sample;
        if (motion_frames > 0 && f + motion_frames >= frames) {
            uint mf = f - (frames - motion_frames);
            motion[((c * motion_frames + mf) * pixels) + p] = sample;
        }
    }
}
)msl";
}
