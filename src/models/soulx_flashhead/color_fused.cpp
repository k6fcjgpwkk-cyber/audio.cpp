#include "color_fused.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace engine::models::soulx_flashhead {
namespace {

float bf16(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

float mad(float x, float scale, float bias) {
    return fmaf(x, scale, bias);
}

float mul(float a, float b) {
    return a * b;
}

float add(float a, float b) {
    return a + b;
}

float sub(float a, float b) {
    return a - b;
}

float divf(float a, float b) {
    return a / b;
}

float clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

float power_f(float x, float exponent) {
    if (exponent == 3.F)
        return mul(mul(x, x), x);
    const float safe = clamp(x, 1.e-30F, 1.e30F);
    return expf(mul(logf(safe), exponent));
}

float step_gt(float x, float bound) {
    return mad(x, 1.F, -bound) > 0.F ? 1.F : 0.F;
}

float select1(float mask, float yes, float no) {
    const float inv = mad(mask, -1.F, 1.F);
    return add(mul(mask, yes), mul(inv, no));
}

void mat3(const float * m, float r, float g, float b, float & x, float & y, float & z) {
    // m[i * 3 + j] is input channel i, output channel j. Accumulation matches a
    // length-3 ggml_vec_dot: fma from zero in input-channel order.
    x = 0.F;
    y = 0.F;
    z = 0.F;
    x = fmaf(r, m[0], x);
    x = fmaf(g, m[3], x);
    x = fmaf(b, m[6], x);
    y = fmaf(r, m[1], y);
    y = fmaf(g, m[4], y);
    y = fmaf(b, m[7], y);
    z = fmaf(r, m[2], z);
    z = fmaf(g, m[5], z);
    z = fmaf(b, m[8], z);
}

void rgb01_to_lab(const ColorMathConstants & k, float r, float g, float b, float & l, float & a, float & bb) {
    const float rg = power_f(mad(r, k.inv_1055, k.bias_1055), k.gamma_exp);
    const float gg = power_f(mad(g, k.inv_1055, k.bias_1055), k.gamma_exp);
    const float bg = power_f(mad(b, k.inv_1055, k.bias_1055), k.gamma_exp);
    const float rl = mul(r, k.inv_1292);
    const float gl = mul(g, k.inv_1292);
    const float bl = mul(b, k.inv_1292);
    const float mr = step_gt(r, k.srgb_knee);
    const float mg = step_gt(g, k.srgb_knee);
    const float mb = step_gt(b, k.srgb_knee);
    const float lr = select1(mr, rg, rl);
    const float lg = select1(mg, gg, gl);
    const float lb = select1(mb, bg, bl);
    float x, y, z;
    mat3(k.rgb_to_xyz, lr, lg, lb, x, y, z);
    x = clamp(divf(x, k.white[0]), k.xyz_min, 1.F);
    y = clamp(divf(y, k.white[1]), k.xyz_min, 1.F);
    z = clamp(divf(z, k.white[2]), k.xyz_min, 1.F);
    const float fx = select1(step_gt(x, k.lab_delta), power_f(x, k.cbrt_exp), mad(x, k.lab_lin_scale, k.lab_lin_bias));
    const float fy = select1(step_gt(y, k.lab_delta), power_f(y, k.cbrt_exp), mad(y, k.lab_lin_scale, k.lab_lin_bias));
    const float fz = select1(step_gt(z, k.lab_delta), power_f(z, k.cbrt_exp), mad(z, k.lab_lin_scale, k.lab_lin_bias));
    l = mad(fy, k.l_scale, k.l_bias);
    a = mul(sub(fx, fy), k.a_scale);
    bb = mul(sub(fy, fz), k.b_scale);
}

void lab_to_rgb01(const ColorMathConstants & k, float l, float a, float b, float & r, float & g, float & bb) {
    const float fy = mad(l, k.inv_l_scale, k.inv_l_bias);
    const float fx = add(mul(a, k.inv_a_scale), fy);
    const float fz = sub(fy, mul(b, k.inv_b_scale));
    const float cube_x = mul(mul(fx, fx), fx);
    const float cube_y = mul(mul(fy, fy), fy);
    const float cube_z = mul(mul(fz, fz), fz);
    const float x = select1(step_gt(cube_x, k.cube_knee), cube_x, mad(fx, k.cube_lin_scale, k.cube_lin_bias));
    const float y = select1(step_gt(l, k.y_knee), cube_y, mul(l, k.y_lin_scale));
    const float z = select1(step_gt(cube_z, k.cube_knee), cube_z, mad(fz, k.cube_lin_scale, k.cube_lin_bias));
    const float xw = mul(x, k.white[0]);
    const float yw = mul(y, k.white[1]);
    const float zw = mul(z, k.white[2]);
    float rr, gg, bch;
    mat3(k.xyz_to_rgb, xw, yw, zw, rr, gg, bch);
    const float er = select1(step_gt(rr, k.enc_knee), mad(power_f(rr, k.enc_exp), k.enc_scale, k.enc_bias),
        mul(rr, k.enc_lin_scale));
    const float eg = select1(step_gt(gg, k.enc_knee), mad(power_f(gg, k.enc_exp), k.enc_scale, k.enc_bias),
        mul(gg, k.enc_lin_scale));
    const float eb = select1(step_gt(bch, k.enc_knee), mad(power_f(bch, k.enc_exp), k.enc_scale, k.enc_bias),
        mul(bch, k.enc_lin_scale));
    r = clamp(er, 0.F, 1.F);
    g = clamp(eg, 0.F, 1.F);
    bb = clamp(eb, 0.F, 1.F);
}

template <typename Fn>
void parallel_for(int64_t count, int threads, Fn && fn) {
    if (count <= 0)
        return;
    threads = std::max(1, std::min(threads, static_cast<int>(count)));
    if (threads == 1) {
        fn(int64_t{0}, count);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int64_t begin = count * t / threads;
        const int64_t end = count * (t + 1) / threads;
        pool.emplace_back([&, begin, end] { fn(begin, end); });
    }
    for (auto & worker : pool)
        worker.join();
}

size_t pixel_index(int64_t frames, int64_t height, int64_t width, int64_t c, int64_t t, int64_t p) {
    return static_cast<size_t>(((c * frames + t) * height * width) + p);
}

float kahan_add(float sum, float & comp, float value) {
    // Separate IEEE adds. volatile stops the host compiler from contracting
    // (value - comp) + sum into an fma, which the Metal kernel does not do
    // when fast math is disabled.
    volatile float y = value - comp;
    volatile float t = sum + y;
    volatile float recovered = t - sum;
    comp = recovered - y;
    return t;
}

void two_sum(float a, float b, float & sum, float & error) {
    // Knuth/Dekker exact sum of two floats. Same expression sequence as the shader.
    volatile float s = a + b;
    volatile float b_virtual = s - a;
    volatile float a_virtual = s - b_virtual;
    error = (a - a_virtual) + (b - b_virtual);
    sum = s;
}

void channel_stats_metal_f32(const float * lab, int64_t frames, int64_t pixels, float * mean, float * stddev) {
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < 3; ++channel) {
            const auto row = color_fused_metal_f32_row(lab + static_cast<size_t>((frame * 3 + channel) * pixels), pixels);
            mean[frame * 3 + channel] = row.mean;
            stddev[frame * 3 + channel] = row.stddev;
        }
    }
}

void channel_stats(const float * lab, int64_t frames, int64_t pixels, float * mean, float * stddev) {
    // Match ggml_compute_forward_mean_f32: double sum, store as float, then
    // divide by float(count). One thread, raster order, so the reduction tree
    // matches the CPU graph.
    for (int64_t f = 0; f < frames; ++f) {
        for (int c = 0; c < 3; ++c) {
            const float * row = lab + static_cast<size_t>((f * 3 + c) * pixels);
            double sum = 0;
            for (int64_t p = 0; p < pixels; ++p)
                sum += static_cast<double>(row[p]);
            const float avg = static_cast<float>(sum) / static_cast<float>(pixels);
            double squares = 0;
            for (int64_t p = 0; p < pixels; ++p) {
                const float delta = sub(row[p], avg);
                squares += static_cast<double>(mul(delta, delta));
            }
            const float variance = static_cast<float>(squares) / static_cast<float>(pixels);
            mean[f * 3 + c] = avg;
            stddev[f * 3 + c] = sqrtf(variance);
        }
    }
}

void ensure_reference(const ColorMathConstants & k, const float * portrait, int64_t height, int64_t width, int threads,
    ColorReferenceCache & cache, bool & cached, double & lab_ms, ColorFusedReduction reduction) {
    const size_t count = static_cast<size_t>(3 * height * width);
    const bool metal_float = reduction == ColorFusedReduction::MetalFloat;
    const bool hit = cache.valid && cache.metal_float == metal_float && cache.height == height && cache.width == width &&
                     cache.portrait.size() == count &&
                     std::memcmp(cache.portrait.data(), portrait, count * sizeof(float)) == 0;
    cached = hit;
    if (hit)
        return;
    const auto started = std::chrono::steady_clock::now();
    const int64_t pixels = height * width;
    std::vector<float> lab(static_cast<size_t>(3 * pixels));
    parallel_for(pixels, threads, [&](int64_t begin, int64_t end) {
        for (int64_t p = begin; p < end; ++p) {
            const int64_t h = p / width;
            const int64_t w = p - h * width;
            const size_t base = static_cast<size_t>(h * width + w);
            const float r = mad(portrait[base], 0.5F, 0.5F);
            const float g = mad(portrait[static_cast<size_t>(pixels) + base], 0.5F, 0.5F);
            const float b = mad(portrait[static_cast<size_t>(2 * pixels) + base], 0.5F, 0.5F);
            float l, a, bb;
            rgb01_to_lab(k, r, g, b, l, a, bb);
            lab[static_cast<size_t>(p)] = l;
            lab[static_cast<size_t>(pixels + p)] = a;
            lab[static_cast<size_t>(2 * pixels + p)] = bb;
        }
    });
    // channel_stats expects [frame][channel][pixel]. One frame is already that
    // layout when channels are planar blocks of `pixels`.
    float mean[3], stddev[3];
    if (metal_float)
        channel_stats_metal_f32(lab.data(), 1, pixels, mean, stddev);
    else
        channel_stats(lab.data(), 1, pixels, mean, stddev);
    cache.portrait.assign(portrait, portrait + count);
    cache.height = height;
    cache.width = width;
    std::copy(mean, mean + 3, cache.mean);
    std::copy(stddev, stddev + 3, cache.stddev);
    cache.metal_float = metal_float;
    cache.valid = true;
    lab_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void copy_window(const float * src, float * dst, int64_t src_frames, int64_t dst_frames, int64_t frame_begin,
    int64_t height, int64_t width) {
    const size_t plane = static_cast<size_t>(height * width);
    const size_t bytes = static_cast<size_t>(dst_frames) * plane * sizeof(float);
    for (int c = 0; c < 3; ++c)
        std::memcpy(dst + static_cast<size_t>(c * dst_frames) * plane,
            src + static_cast<size_t>(c * src_frames + frame_begin) * plane, bytes);
}

} // namespace

MetalFloatRowStats color_fused_metal_f32_row(const float * row, int64_t pixels) {
    if (pixels <= 0)
        throw std::runtime_error("FlashHead Metal float stats require a positive row");
    auto reduce_pairs = [](float * hi, float * lo) {
        for (int stride = kMetalStatsLanes / 2; stride > 0; stride >>= 1) {
            for (int lane = 0; lane < stride; ++lane) {
                float sum = 0.F;
                float error = 0.F;
                two_sum(hi[lane], hi[lane + stride], sum, error);
                const float low = (lo[lane] + lo[lane + stride]) + error;
                two_sum(sum, low, hi[lane], lo[lane]);
            }
        }
    };
    float hi[kMetalStatsLanes];
    float lo[kMetalStatsLanes];
    for (int lane = 0; lane < kMetalStatsLanes; ++lane) {
        float sum = 0.F;
        float comp = 0.F;
        for (int64_t p = lane; p < pixels; p += kMetalStatsLanes)
            sum = kahan_add(sum, comp, row[p]);
        hi[lane] = sum;
        lo[lane] = -comp;
    }
    reduce_pairs(hi, lo);
    const float mean = (hi[0] + lo[0]) / static_cast<float>(pixels);
    for (int lane = 0; lane < kMetalStatsLanes; ++lane) {
        float sum = 0.F;
        float comp = 0.F;
        for (int64_t p = lane; p < pixels; p += kMetalStatsLanes) {
            const float delta = row[p] - mean;
            sum = kahan_add(sum, comp, delta * delta);
        }
        hi[lane] = sum;
        lo[lane] = -comp;
    }
    reduce_pairs(hi, lo);
    MetalFloatRowStats stats;
    stats.mean = mean;
    const float variance = std::max(0.F, (hi[0] + lo[0]) / static_cast<float>(pixels));
    stats.stddev = sqrtf(variance);
    return stats;
}

void color_fused_channel_stats(const float * lab, int64_t frames, int64_t pixels, float * mean, float * stddev) {
    channel_stats(lab, frames, pixels, mean, stddev);
}

const ColorMathConstants & color_math_constants() {
    static const ColorMathConstants constants = [] {
        ColorMathConstants k{};
        const float rgb[] = {.4124564F, .2126729F, .0193339F, .3575761F, .7151522F, .119192F, .1804375F, .072175F,
            .9503041F};
        const float xyz[] = {3.2404542F, -.969266F, .0556434F, -1.5371385F, 1.8760108F, -.2040259F, -.4985314F,
            .041556F, 1.0572252F};
        for (int i = 0; i < 9; ++i) {
            k.rgb_to_xyz[i] = bf16(rgb[i]);
            k.xyz_to_rgb[i] = bf16(xyz[i]);
        }
        // Already the bf16 values stored by the graph, not a fresh rounding of D65.
        k.white[0] = .94921875F;
        k.white[1] = 1.F;
        k.white[2] = 1.0859375F;
        k.std_floor = bf16(1.e-8F);
        k.inv_1055 = 1.F / 1.055F;
        k.bias_1055 = .055F / 1.055F;
        k.gamma_exp = bf16(2.4F);
        k.inv_1292 = 1.F / 12.92F;
        k.srgb_knee = bf16(.04045F);
        k.lab_delta = bf16(.008856F);
        k.lab_lin_scale = 903.3F / 116.F;
        k.lab_lin_bias = 16.F / 116.F;
        k.xyz_min = bf16(1.e-8F);
        k.cbrt_exp = bf16(1.F / 3.F);
        k.l_scale = 116.F;
        k.l_bias = -16.F;
        k.a_scale = 500.F;
        k.b_scale = 200.F;
        k.inv_l_scale = 1.F / 116.F;
        k.inv_l_bias = 16.F / 116.F;
        k.inv_a_scale = 1.F / 500.F;
        k.inv_b_scale = 1.F / 200.F;
        k.cube_knee = bf16(.008856F);
        k.y_knee = bf16(903.3F * .008856F);
        k.y_lin_scale = 1.F / 903.3F;
        k.cube_lin_scale = 116.F / 903.3F;
        k.cube_lin_bias = -16.F / 903.3F;
        k.enc_exp = bf16(1.F / 2.4F);
        k.enc_scale = 1.055F;
        k.enc_bias = -.055F;
        k.enc_knee = bf16(.0031308F);
        k.enc_lin_scale = 12.92F;
        return k;
    }();
    return constants;
}

bool color_fused_metal_compiled() {
#if defined(SOULX_FLASHHEAD_COLOR_METAL)
    return true;
#else
    return false;
#endif
}

void color_fused_apply_cpu(const float * video, const float * portrait, int64_t height, int64_t width,
    int64_t frame_begin, int64_t frame_count, int64_t motion_frames, float strength, int threads,
    ColorReferenceCache & cache, float * video_out, float * motion_out, ColorFusedTiming & timing,
    ColorFusedReduction reduction) {
    const auto wall = std::chrono::steady_clock::now();
    timing = {};
    timing.backend = "cpu";
    if (height <= 0 || width <= 0 || frame_begin < 0 || frame_count <= 0 || motion_frames < 0 ||
        motion_frames > frame_count || frame_begin + frame_count > 33)
        throw std::runtime_error("FlashHead fused color window is out of range");
    if (strength == 0.F) {
        copy_window(video, video_out, 33, frame_count, frame_begin, height, width);
        if (motion_frames > 0)
            copy_window(video, motion_out, 33, motion_frames, frame_begin + frame_count - motion_frames, height, width);
        timing.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count();
        debug::timing_log_scalar("soulx_flashhead.color.impl", std::string_view("fused_copy"));
        debug::timing_log_scalar("soulx_flashhead.color.wall_ms", timing.wall_ms);
        return;
    }
    const auto & k = color_math_constants();
    threads = std::max(1, threads);
    ensure_reference(k, portrait, height, width, threads, cache, timing.reference_cached, timing.lab_ms, reduction);
    const int64_t pixels = height * width;
    std::vector<float> lab(static_cast<size_t>(frame_count * 3 * pixels));
    const auto lab_started = std::chrono::steady_clock::now();
    parallel_for(frame_count * pixels, threads, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t f = i / pixels;
            const int64_t p = i - f * pixels;
            const int64_t t = frame_begin + f;
            float rgb[3];
            for (int c = 0; c < 3; ++c)
                rgb[c] = mad(video[pixel_index(33, height, width, c, t, p)], 0.5F, 0.5F);
            float l, a, b;
            rgb01_to_lab(k, rgb[0], rgb[1], rgb[2], l, a, b);
            float * row = lab.data() + static_cast<size_t>(f * 3 * pixels + p);
            row[0] = l;
            row[static_cast<size_t>(pixels)] = a;
            row[static_cast<size_t>(2 * pixels)] = b;
        }
    });
    timing.lab_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lab_started).count();
    std::vector<float> mean(static_cast<size_t>(frame_count * 3));
    std::vector<float> stddev(static_cast<size_t>(frame_count * 3));
    // Repack to [frame][channel][pixel] for the sequential reduction. The pixel
    // loop above stored [frame][pixel] with channel planes, which is the same
    // [frame][channel][pixel] order: channel c of frame f is at f*3*pixels + c*pixels.
    const auto stats_started = std::chrono::steady_clock::now();
    if (reduction == ColorFusedReduction::MetalFloat)
        channel_stats_metal_f32(lab.data(), frame_count, pixels, mean.data(), stddev.data());
    else
        channel_stats(lab.data(), frame_count, pixels, mean.data(), stddev.data());
    std::vector<float> ratio(static_cast<size_t>(frame_count * 3));
    for (int64_t f = 0; f < frame_count; ++f) {
        for (int c = 0; c < 3; ++c) {
            const float dev = stddev[static_cast<size_t>(f * 3 + c)];
            const float safe = dev < k.std_floor ? 1.F : dev;
            ratio[static_cast<size_t>(f * 3 + c)] = divf(cache.stddev[c], safe);
        }
    }
    timing.stats_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stats_started).count();
    const float inverse = mad(strength, -1.F, 1.F);
    const auto match_started = std::chrono::steady_clock::now();
    parallel_for(frame_count * pixels, threads, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t f = i / pixels;
            const int64_t p = i - f * pixels;
            const int64_t t = frame_begin + f;
            const float * base = lab.data() + static_cast<size_t>(f * 3 * pixels + p);
            float corrected[3];
            for (int c = 0; c < 3; ++c) {
                const float centered = sub(c == 0 ? base[0] : base[static_cast<size_t>(c * pixels)], mean[static_cast<size_t>(f * 3 + c)]);
                const float scaled = mul(centered, ratio[static_cast<size_t>(f * 3 + c)]);
                corrected[c] = add(scaled, cache.mean[c]);
            }
            float rr, gg, bb;
            lab_to_rgb01(k, corrected[0], corrected[1], corrected[2], rr, gg, bb);
            const float corrected_rgb[3] = {rr, gg, bb};
            for (int c = 0; c < 3; ++c) {
                const float source = mad(video[pixel_index(33, height, width, c, t, p)], 0.5F, 0.5F);
                const float blended = add(mul(source, inverse), mul(corrected_rgb[c], strength));
                const float sample = mad(blended, 2.F, -1.F);
                video_out[pixel_index(frame_count, height, width, c, f, p)] = sample;
                if (motion_out != nullptr && f >= frame_count - motion_frames) {
                    const int64_t mf = f - (frame_count - motion_frames);
                    motion_out[pixel_index(motion_frames, height, width, c, mf, p)] = sample;
                }
            }
        }
    });
    timing.match_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - match_started).count();
    timing.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count();
    debug::timing_log_scalar("soulx_flashhead.color.impl",
        std::string_view(reduction == ColorFusedReduction::MetalFloat ? "fused_cpu_metal_f32_stats" : "fused_cpu"));
    debug::timing_log_scalar("soulx_flashhead.color.reduction",
        std::string_view(reduction == ColorFusedReduction::MetalFloat ? "metal_f32" : "double"));
    debug::timing_log_scalar("soulx_flashhead.color.lab_ms", timing.lab_ms);
    debug::timing_log_scalar("soulx_flashhead.color.stats_ms", timing.stats_ms);
    debug::timing_log_scalar("soulx_flashhead.color.match_ms", timing.match_ms);
    debug::timing_log_scalar("soulx_flashhead.color.reference_cached", timing.reference_cached);
    debug::timing_log_scalar("soulx_flashhead.color.threads", static_cast<int64_t>(threads));
    debug::timing_log_scalar("soulx_flashhead.color.wall_ms", timing.wall_ms);
}

bool color_fused_execute_device(core::ExecutionContext & execution, const ColorFusedDeviceBuffers & buffers,
    const float * portrait, int64_t height, int64_t width, int64_t frame_begin, int64_t frame_count,
    int64_t motion_frames, float strength, int threads, ColorReferenceCache & cache) {
    if (buffers.video_in == nullptr || buffers.video_out == nullptr ||
        (motion_frames > 0 && buffers.motion_out == nullptr))
        throw std::runtime_error("FlashHead fused color is missing device buffers");
    const size_t input_count = static_cast<size_t>(3 * 33 * height * width);
    if (static_cast<size_t>(ggml_nelements(buffers.video_in)) != input_count || !ggml_is_contiguous(buffers.video_in) ||
        buffers.video_in->type != GGML_TYPE_F32)
        throw std::runtime_error("FlashHead fused color expects a contiguous F32 33-frame video");
    auto read = [](ggml_tensor * tensor, std::vector<float> & values) {
        values.resize(static_cast<size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    };
    auto write = [](ggml_tensor * tensor, const std::vector<float> & values) {
        if (static_cast<size_t>(ggml_nelements(tensor)) != values.size())
            throw std::runtime_error("FlashHead fused color output size mismatch");
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    };
    if (strength == 0.F) {
        std::vector<float> input, output, motion;
        read(buffers.video_in, input);
        output.resize(static_cast<size_t>(3 * frame_count * height * width));
        if (motion_frames > 0)
            motion.resize(static_cast<size_t>(3 * motion_frames * height * width));
        ColorFusedTiming timing;
        color_fused_apply_cpu(input.data(), portrait, height, width, frame_begin, frame_count, motion_frames, 0.F,
            threads, cache, output.data(), motion_frames > 0 ? motion.data() : nullptr, timing);
        write(buffers.video_out, output);
        if (motion_frames > 0)
            write(buffers.motion_out, motion);
        return true;
    }
    if (execution.backend_type() == core::BackendType::Cpu) {
        if (buffers.video_in->data == nullptr || buffers.video_out->data == nullptr)
            throw std::runtime_error("FlashHead fused color CPU tensors are not addressable");
        ColorFusedTiming timing;
        color_fused_apply_cpu(static_cast<const float *>(buffers.video_in->data), portrait, height, width, frame_begin,
            frame_count, motion_frames, strength, threads, cache, static_cast<float *>(buffers.video_out->data),
            motion_frames > 0 ? static_cast<float *>(buffers.motion_out->data) : nullptr, timing);
        return true;
    }
#if defined(SOULX_FLASHHEAD_COLOR_METAL)
    if (execution.backend_type() == core::BackendType::Metal) {
        if (!color_fused_metal_apply(execution, buffers, portrait, height, width, frame_begin, frame_count,
                motion_frames, strength, cache))
            return false;
        return true;
    }
#else
    (void)execution;
#endif
    return false;
}

} // namespace engine::models::soulx_flashhead
