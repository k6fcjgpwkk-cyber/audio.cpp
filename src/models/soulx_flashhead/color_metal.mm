#include "color_fused.h"

#include "engine/framework/debug/trace.h"
#include "ggml-metal-device.h"

#include <ggml-backend.h>

#include <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::soulx_flashhead {
namespace {

// Constant-float indices shared with the embedded shader. Keeping them flat
// avoids Metal/C++ struct padding differences.
enum Constant {
    kRgb = 0,
    kXyz = 9,
    kWhite = 18,
    kInv1055 = 21,
    kBias1055 = 22,
    kGamma = 23,
    kInv1292 = 24,
    kSrgbKnee = 25,
    kLabDelta = 26,
    kLabLinScale = 27,
    kLabLinBias = 28,
    kXyzMin = 29,
    kCbrt = 30,
    kLScale = 31,
    kLBias = 32,
    kAScale = 33,
    kBScale = 34,
    kInvLScale = 35,
    kInvLBias = 36,
    kInvA = 37,
    kInvB = 38,
    kCubeKnee = 39,
    kYKnee = 40,
    kYLin = 41,
    kCubeLinScale = 42,
    kCubeLinBias = 43,
    kEncExp = 44,
    kEncScale = 45,
    kEncBias = 46,
    kEncKnee = 47,
    kEncLin = 48,
    kStrength = 49,
    kInverse = 50,
    kRefMean = 51,
    kRefStd = 54,
    kStdFloor = 57,
    kConstantCount = 58,
};

enum Uniform {
    kWidth = 0,
    kHeight = 1,
    kInFrames = 2,
    kFrameBegin = 3,
    kFrameCount = 4,
    kPixels = 5,
    kSrcLo = 6,
    kSrcHi = 7,
    kDstLo = 8,
    kDstHi = 9,
    kMotionLo = 10,
    kMotionHi = 11,
    kMotionFrames = 12,
    kUniformCount = 13,
};

constexpr const char * kShader = R"msl(
#include <metal_stdlib>
using namespace metal;
// MSL 4.1 section 2.1: the language has no double, long double, or 64-bit float.
// Every value in this shader is float (32-bit). The host must compile it with
// fast math disabled. Fast math reassociates adds and contracts them into fma,
// which would change the Kahan compensation below (MSL spec 1.6.3).
// flashhead_color_stats matches color_fused_metal_f32_row: 256 lanes, Kahan in
// increasing index order, fold sum-comp, then a float tree of strides 128..1.

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

struct Pipelines {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> lab = nil;
    id<MTLComputePipelineState> stats = nil;
    id<MTLComputePipelineState> match = nil;
    ~Pipelines() {
        if (match != nil)
            [match release];
        if (stats != nil)
            [stats release];
        if (lab != nil)
            [lab release];
        if (queue != nil)
            [queue release];
        if (device != nil)
            [device release];
    }
};

Pipelines & pipelines_for(id<MTLDevice> device) {
    static Pipelines state;
    if (state.device == device && state.lab != nil && state.stats != nil && state.match != nil)
        return state;
    if (state.match != nil)
        [state.match release];
    if (state.stats != nil)
        [state.stats release];
    if (state.lab != nil)
        [state.lab release];
    if (state.queue != nil)
        [state.queue release];
    if (state.device != nil)
        [state.device release];
    state = {};
    state.device = [device retain];
    state.queue = [device newCommandQueue];
    if (state.queue == nil)
        throw std::runtime_error("FlashHead fused color cannot create a Metal command queue");
    MTLCompileOptions * options = [[MTLCompileOptions alloc] init];
    // MSL's default math mode is fast: it may reassociate and contract adds.
    // The Kahan stats kernel needs the IEEE add/sub sequence. fastMathEnabled
    // is the switch available since the first Metal SDK. mathMode=Safe is the
    // same contract on macOS 15 and later (MSL spec 1.6.3).
    options.fastMathEnabled = NO;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *))
        options.mathMode = MTLMathModeSafe;
#endif
    NSError * error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kShader] options:options error:&error];
    [options release];
    if (library == nil) {
        const char * message = error != nil ? [[error localizedDescription] UTF8String] : "unknown Metal compile error";
        throw std::runtime_error(std::string("FlashHead fused color Metal compile failed: ") + message);
    }
    id<MTLFunction> lab = [library newFunctionWithName:@"flashhead_color_lab"];
    id<MTLFunction> stats = [library newFunctionWithName:@"flashhead_color_stats"];
    id<MTLFunction> match = [library newFunctionWithName:@"flashhead_color_match"];
    if (lab == nil || stats == nil || match == nil) {
        [library release];
        throw std::runtime_error("FlashHead fused color Metal shader is missing a kernel");
    }
    state.lab = [device newComputePipelineStateWithFunction:lab error:&error];
    state.stats = [device newComputePipelineStateWithFunction:stats error:&error];
    state.match = [device newComputePipelineStateWithFunction:match error:&error];
    [lab release];
    [stats release];
    [match release];
    [library release];
    if (state.lab == nil || state.stats == nil || state.match == nil) {
        const char * message = error != nil ? [[error localizedDescription] UTF8String] : "unknown pipeline error";
        throw std::runtime_error(std::string("FlashHead fused color Metal pipeline failed: ") + message);
    }
    return state;
}

struct Binding {
    id<MTLBuffer> buffer = nil;
    uint64_t offset = 0;
};

Binding bind_tensor(const ggml_tensor * tensor, const char * what) {
    if (tensor == nullptr || tensor->buffer == nullptr)
        throw std::runtime_error(std::string("FlashHead fused color Metal tensor has no buffer: ") + what);
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || buffer->context == nullptr)
        throw std::runtime_error(std::string("FlashHead fused color Metal buffer is missing: ") + what);
    const ggml_metal_buffer_id id = ggml_metal_buffer_get_id(static_cast<ggml_metal_buffer_t>(buffer->context), tensor);
    if (id.metal == nil)
        throw std::runtime_error(std::string("FlashHead fused color cannot find the Metal buffer for ") + what);
    if ((id.offs & 3u) != 0)
        throw std::runtime_error(std::string("FlashHead fused color Metal offset is not float-aligned: ") + what);
    Binding binding;
    binding.buffer = (id<MTLBuffer>)id.metal;
    binding.offset = static_cast<uint64_t>(id.offs);
    return binding;
}

void put_offset(uint32_t * uniform, int lo, int hi, uint64_t offset) {
    uniform[lo] = static_cast<uint32_t>(offset);
    uniform[hi] = static_cast<uint32_t>(offset >> 32);
}

void fill_constants(float * dst, float strength) {
    const auto & k = color_math_constants();
    std::memset(dst, 0, sizeof(float) * kConstantCount);
    std::memcpy(dst + kRgb, k.rgb_to_xyz, sizeof(k.rgb_to_xyz));
    std::memcpy(dst + kXyz, k.xyz_to_rgb, sizeof(k.xyz_to_rgb));
    std::memcpy(dst + kWhite, k.white, sizeof(k.white));
    dst[kInv1055] = k.inv_1055;
    dst[kBias1055] = k.bias_1055;
    dst[kGamma] = k.gamma_exp;
    dst[kInv1292] = k.inv_1292;
    dst[kSrgbKnee] = k.srgb_knee;
    dst[kLabDelta] = k.lab_delta;
    dst[kLabLinScale] = k.lab_lin_scale;
    dst[kLabLinBias] = k.lab_lin_bias;
    dst[kXyzMin] = k.xyz_min;
    dst[kCbrt] = k.cbrt_exp;
    dst[kLScale] = k.l_scale;
    dst[kLBias] = k.l_bias;
    dst[kAScale] = k.a_scale;
    dst[kBScale] = k.b_scale;
    dst[kInvLScale] = k.inv_l_scale;
    dst[kInvLBias] = k.inv_l_bias;
    dst[kInvA] = k.inv_a_scale;
    dst[kInvB] = k.inv_b_scale;
    dst[kCubeKnee] = k.cube_knee;
    dst[kYKnee] = k.y_knee;
    dst[kYLin] = k.y_lin_scale;
    dst[kCubeLinScale] = k.cube_lin_scale;
    dst[kCubeLinBias] = k.cube_lin_bias;
    dst[kEncExp] = k.enc_exp;
    dst[kEncScale] = k.enc_scale;
    dst[kEncBias] = k.enc_bias;
    dst[kEncKnee] = k.enc_knee;
    dst[kEncLin] = k.enc_lin_scale;
    dst[kStrength] = strength;
    dst[kInverse] = fmaf(strength, -1.F, 1.F);
    dst[kStdFloor] = k.std_floor;
}

void encode_stats(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline, id<MTLBuffer> lab_buffer,
    id<MTLBuffer> stats_buffer, const uint32_t * uniform, NSUInteger uniform_bytes, NSUInteger groups) {
    if ([pipeline maxTotalThreadsPerThreadgroup] < 256)
        throw std::runtime_error("FlashHead fused color stats kernel requires 256 threads in a group");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:lab_buffer offset:0 atIndex:0];
    [encoder setBuffer:stats_buffer offset:0 atIndex:1];
    [encoder setBytes:uniform length:uniform_bytes atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
}

} // namespace

bool color_fused_metal_apply(core::ExecutionContext & execution, const ColorFusedDeviceBuffers & buffers,
    const float * portrait, int64_t height, int64_t width, int64_t frame_begin, int64_t frame_count,
    int64_t motion_frames, float strength, ColorReferenceCache & cache) {
    if (buffers.lab == nullptr || buffers.portrait == nullptr || buffers.portrait_lab == nullptr ||
        buffers.motion_out == nullptr || buffers.stats == nullptr || buffers.portrait_stats == nullptr)
        throw std::runtime_error("FlashHead fused color Metal scratch buffers were not allocated");
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const auto wall = std::chrono::steady_clock::now();
    ggml_backend_synchronize(execution.backend());
    const Binding video = bind_tensor(buffers.video_in, "video");
    const Binding lab = bind_tensor(buffers.lab, "lab");
    const Binding portrait_buf = bind_tensor(buffers.portrait, "portrait");
    const Binding portrait_lab = bind_tensor(buffers.portrait_lab, "portrait lab");
    const Binding output = bind_tensor(buffers.video_out, "output");
    const Binding motion = bind_tensor(buffers.motion_out, "motion");
    const Binding stats = bind_tensor(buffers.stats, "stats");
    const Binding portrait_stats = bind_tensor(buffers.portrait_stats, "portrait stats");
    if (video.buffer.device != lab.buffer.device)
        throw std::runtime_error("FlashHead fused color Metal buffers are on different devices");
    Pipelines & pipes = pipelines_for(video.buffer.device);
    float constants[kConstantCount];
    fill_constants(constants, strength);
    const int64_t pixels = height * width;
    const size_t portrait_bytes = static_cast<size_t>(3 * pixels) * sizeof(float);
    const bool reference_cached = cache.valid && cache.metal_float && cache.height == height && cache.width == width &&
                                  cache.portrait.size() == static_cast<size_t>(3 * pixels) &&
                                  std::memcmp(cache.portrait.data(), portrait, portrait_bytes) == 0;
    double lab_ms = 0;
    if (!reference_cached) {
        ggml_backend_tensor_set(buffers.portrait, portrait, 0, portrait_bytes);
        cache.portrait.assign(portrait, portrait + 3 * pixels);
        cache.height = height;
        cache.width = width;
        uint32_t uniform[kUniformCount] = {};
        uniform[kWidth] = static_cast<uint32_t>(width);
        uniform[kHeight] = static_cast<uint32_t>(height);
        uniform[kInFrames] = 1;
        uniform[kFrameCount] = 1;
        uniform[kPixels] = static_cast<uint32_t>(pixels);
        put_offset(uniform, kSrcLo, kSrcHi, portrait_buf.offset);
        put_offset(uniform, kDstLo, kDstHi, portrait_lab.offset);
        const auto started = std::chrono::steady_clock::now();
        @autoreleasepool {
            id<MTLCommandBuffer> command = [pipes.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipes.lab];
            [encoder setBuffer:portrait_buf.buffer offset:0 atIndex:0];
            [encoder setBuffer:portrait_lab.buffer offset:0 atIndex:1];
            [encoder setBytes:uniform length:sizeof(uniform) atIndex:2];
            [encoder setBytes:constants length:sizeof(constants) atIndex:3];
            NSUInteger width_threads = std::max<NSUInteger>(1, std::min<NSUInteger>(256, [pipes.lab maxTotalThreadsPerThreadgroup]));
            NSUInteger groups = (static_cast<NSUInteger>(pixels) + width_threads - 1) / width_threads;
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(width_threads, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if ([command status] != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("FlashHead fused color reference Lab dispatch failed");
        }
        uint32_t stats_uniform[6] = {};
        stats_uniform[0] = static_cast<uint32_t>(pixels);
        stats_uniform[1] = 1;
        put_offset(stats_uniform, 2, 3, portrait_lab.offset);
        put_offset(stats_uniform, 4, 5, portrait_stats.offset);
        const auto stats_started = std::chrono::steady_clock::now();
        @autoreleasepool {
            id<MTLCommandBuffer> command = [pipes.queue commandBuffer];
            encode_stats(command, pipes.stats, portrait_lab.buffer, portrait_stats.buffer, stats_uniform,
                sizeof(stats_uniform), 3);
            [command commit];
            [command waitUntilCompleted];
            if ([command status] != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("FlashHead fused color reference stats dispatch failed");
        }
        lab_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        float reduced[6];
        ggml_backend_tensor_get(buffers.portrait_stats, reduced, 0, sizeof(reduced));
        std::memcpy(cache.mean, reduced, sizeof(float) * 3);
        std::memcpy(cache.stddev, reduced + 3, sizeof(float) * 3);
        cache.metal_float = true;
        cache.valid = true;
        (void)stats_started;
    }
    uint32_t uniform[20] = {};
    uniform[kWidth] = static_cast<uint32_t>(width);
    uniform[kHeight] = static_cast<uint32_t>(height);
    uniform[kInFrames] = 33;
    uniform[kFrameBegin] = static_cast<uint32_t>(frame_begin);
    uniform[kFrameCount] = static_cast<uint32_t>(frame_count);
    uniform[kPixels] = static_cast<uint32_t>(pixels);
    put_offset(uniform, kSrcLo, kSrcHi, video.offset);
    put_offset(uniform, kDstLo, kDstHi, lab.offset);
    const auto lab_started = std::chrono::steady_clock::now();
    @autoreleasepool {
        id<MTLCommandBuffer> command = [pipes.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipes.lab];
        [encoder setBuffer:video.buffer offset:0 atIndex:0];
        [encoder setBuffer:lab.buffer offset:0 atIndex:1];
        [encoder setBytes:uniform length:sizeof(uint32_t) * kUniformCount atIndex:2];
        [encoder setBytes:constants length:sizeof(constants) atIndex:3];
        NSUInteger width_threads = std::max<NSUInteger>(1, std::min<NSUInteger>(256, [pipes.lab maxTotalThreadsPerThreadgroup]));
        const NSUInteger count = static_cast<NSUInteger>(frame_count * pixels);
        NSUInteger groups = (count + width_threads - 1) / width_threads;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(width_threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("FlashHead fused color video Lab dispatch failed");
    }
    lab_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lab_started).count();
    uint32_t stats_uniform[6] = {};
    stats_uniform[0] = static_cast<uint32_t>(pixels);
    stats_uniform[1] = static_cast<uint32_t>(frame_count);
    put_offset(stats_uniform, 2, 3, lab.offset);
    put_offset(stats_uniform, 4, 5, stats.offset);
    const auto stats_started = std::chrono::steady_clock::now();
    @autoreleasepool {
        id<MTLCommandBuffer> command = [pipes.queue commandBuffer];
        encode_stats(command, pipes.stats, lab.buffer, stats.buffer, stats_uniform, sizeof(stats_uniform),
            static_cast<NSUInteger>(frame_count * 3));
        [command commit];
        [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("FlashHead fused color stats dispatch failed");
    }
    const double stats_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stats_started).count();
    std::memcpy(constants + kRefMean, cache.mean, sizeof(float) * 3);
    std::memcpy(constants + kRefStd, cache.stddev, sizeof(float) * 3);
    uint32_t match_uniform[20] = {};
    match_uniform[kWidth] = static_cast<uint32_t>(width);
    match_uniform[kHeight] = static_cast<uint32_t>(height);
    match_uniform[kInFrames] = 33;
    match_uniform[kFrameBegin] = static_cast<uint32_t>(frame_begin);
    match_uniform[kFrameCount] = static_cast<uint32_t>(frame_count);
    match_uniform[kPixels] = static_cast<uint32_t>(pixels);
    put_offset(match_uniform, kSrcLo, kSrcHi, video.offset);
    put_offset(match_uniform, kDstLo, kDstHi, lab.offset);
    put_offset(match_uniform, 14, 15, output.offset);
    put_offset(match_uniform, 16, 17, motion.offset);
    put_offset(match_uniform, 18, 19, stats.offset);
    match_uniform[kMotionFrames] = static_cast<uint32_t>(motion_frames);
    const auto match_started = std::chrono::steady_clock::now();
    @autoreleasepool {
        id<MTLCommandBuffer> command = [pipes.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipes.match];
        [encoder setBuffer:video.buffer offset:0 atIndex:0];
        [encoder setBuffer:lab.buffer offset:0 atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBuffer:motion.buffer offset:0 atIndex:3];
        [encoder setBytes:match_uniform length:sizeof(match_uniform) atIndex:4];
        [encoder setBytes:constants length:sizeof(constants) atIndex:5];
        [encoder setBuffer:stats.buffer offset:0 atIndex:6];
        NSUInteger width_threads = std::max<NSUInteger>(1, std::min<NSUInteger>(256, [pipes.match maxTotalThreadsPerThreadgroup]));
        const NSUInteger count = static_cast<NSUInteger>(frame_count * pixels);
        NSUInteger groups = (count + width_threads - 1) / width_threads;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(width_threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("FlashHead fused color match dispatch failed");
    }
    const double match_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - match_started).count();
    ggml_backend_synchronize(execution.backend());
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count();
    debug::timing_log_scalar("soulx_flashhead.color.impl", std::string_view("fused_metal"));
    debug::timing_log_scalar("soulx_flashhead.color.reduction", std::string_view("metal_f32"));
    debug::timing_log_scalar("soulx_flashhead.color.lab_ms", lab_ms);
    debug::timing_log_scalar("soulx_flashhead.color.stats_ms", stats_ms);
    debug::timing_log_scalar("soulx_flashhead.color.match_ms", match_ms);
    debug::timing_log_scalar("soulx_flashhead.color.reference_cached", reference_cached);
    debug::timing_log_scalar("soulx_flashhead.color.wall_ms", wall_ms);
    return true;
}

} // namespace engine::models::soulx_flashhead
