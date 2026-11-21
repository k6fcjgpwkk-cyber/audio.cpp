#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/soulx_flashhead/color.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_tensor;

namespace engine::models::soulx_flashhead {

struct ColorMathConstants {
    float rgb_to_xyz[9];
    float xyz_to_rgb[9];
    float white[3];
    float std_floor;
    float inv_1055;
    float bias_1055;
    float gamma_exp;
    float inv_1292;
    float srgb_knee;
    float lab_delta;
    float lab_lin_scale;
    float lab_lin_bias;
    float xyz_min;
    float cbrt_exp;
    float l_scale;
    float l_bias;
    float a_scale;
    float b_scale;
    float inv_l_scale;
    float inv_l_bias;
    float inv_a_scale;
    float inv_b_scale;
    float cube_knee;
    float y_knee;
    float y_lin_scale;
    float cube_lin_scale;
    float cube_lin_bias;
    float enc_exp;
    float enc_scale;
    float enc_bias;
    float enc_knee;
    float enc_lin_scale;
};

struct ColorFusedTiming {
    double lab_ms = 0;
    double stats_ms = 0;
    double match_ms = 0;
    double wall_ms = 0;
    bool reference_cached = false;
    const char * backend = "cpu";
};

// Portrait Lab mean/std for one resolution. Reused when the portrait bytes match.
struct ColorReferenceCache {
    std::vector<float> portrait;
    int64_t height = 0;
    int64_t width = 0;
    float mean[3] = {};
    float stddev[3] = {};
    bool valid = false;
    // True when mean/stddev came from the Metal float reduction.
    bool metal_float = false;
};

struct ColorFusedDeviceBuffers {
    ggml_tensor * video_in = nullptr;
    ggml_tensor * video_out = nullptr;
    ggml_tensor * motion_out = nullptr;
    // Metal lab scratch. Null on the CPU path.
    ggml_tensor * lab = nullptr;
    ggml_tensor * portrait = nullptr;
    ggml_tensor * portrait_lab = nullptr;
    // Metal stats kernels write mean[frames*3] then stddev[frames*3]. The host
    // does not reduce those rows.
    ggml_tensor * stats = nullptr;
    ggml_tensor * portrait_stats = nullptr;
};

// Lane count of flashhead_color_stats. The CPU replica and the MSL kernel must
// use the same value. MSL has no double (specification 4.1, section 2.1).
constexpr int kMetalStatsLanes = 256;

enum class ColorFusedReduction {
    // Double accumulation, then store the mean as float. Matches CPU ggml_mean.
    Double,
    // 256-lane Kahan sum and a float tree. This is the Metal stats kernel.
    MetalFloat,
};

struct MetalFloatRowStats {
    float mean = 0;
    float stddev = 0;
};

// One Lab row, length pixels, in the order the stats kernel reads it.
MetalFloatRowStats color_fused_metal_f32_row(const float * row, int64_t pixels);

const ColorMathConstants & color_math_constants();
bool color_fused_metal_compiled();

// lab is [frame][channel][pixel]. Double accumulation matches CPU ggml_mean.
void color_fused_channel_stats(const float * lab, int64_t frames, int64_t pixels, float * mean, float * stddev);

// video is logical {3, 33, height, width} in contiguous ggml order.
// frame_begin/frame_count select the delivered window. motion_out may be null
// when motion_frames is 0. motion_frames is the tail of the delivered window.
// MetalFloat uses the reduction the shader runs. It is not a GPU execution.
void color_fused_apply_cpu(const float * video, const float * portrait, int64_t height, int64_t width,
    int64_t frame_begin, int64_t frame_count, int64_t motion_frames, float strength, int threads,
    ColorReferenceCache & cache, float * video_out, float * motion_out, ColorFusedTiming & timing,
    ColorFusedReduction reduction = ColorFusedReduction::Double);

// Returns false when this backend has no fused implementation in the binary.
bool color_fused_execute_device(core::ExecutionContext & execution, const ColorFusedDeviceBuffers & buffers,
    const float * portrait, int64_t height, int64_t width, int64_t frame_begin, int64_t frame_count,
    int64_t motion_frames, float strength, int threads, ColorReferenceCache & cache);

// Defined only in the macOS Metal translation unit. Do not call unless
// color_fused_metal_compiled() is true.
bool color_fused_metal_apply(core::ExecutionContext & execution, const ColorFusedDeviceBuffers & buffers,
    const float * portrait, int64_t height, int64_t width, int64_t frame_begin, int64_t frame_count,
    int64_t motion_frames, float strength, ColorReferenceCache & cache);

} // namespace engine::models::soulx_flashhead
