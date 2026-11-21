#include "engine/framework/core/backend.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/soulx_flashhead/color.h"
#include "engine/models/soulx_flashhead/exec_plan.h"
#include "color_fused.h"
#include "color_metal_shader.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// CPU graph versus CPU fused on finite inputs. Strength 0 is exact and is not
// covered by these gates. They are not a Metal measurement.
constexpr double kMaxAbsGate = 1e-4;
constexpr double kRmseGate = 2e-5;
// CPU graph versus the float Kahan/two-sum reduction implemented by the Metal
// stats kernel. The comparison uses the CPU replica and libm per-pixel math.
// It does not execute the shader. Observed max_abs on the images below was 0.
constexpr double kMetalFloatMaxAbsGate = 1e-4;
constexpr double kMetalFloatRmseGate = 2e-5;
static_assert(kMetalFloatMaxAbsGate == kMaxAbsGate, "float replica keeps the same max abs gate");
static_assert(kMetalFloatRmseGate == kRmseGate, "float replica keeps the same rmse gate");

struct Error {
    double max_abs = 0;
    double rmse = 0;
    int64_t rgb8_mismatch = 0;
    double rgb8_mae = 0;
    int64_t finite_pairs = 0;
    int64_t nonfinite_actual = 0;
    int64_t nonfinite_expected = 0;
    bool bitwise = false;
    bool have_sample = false;
    bool sample_nonfinite = false;
    size_t sample_index = 0;
    int64_t sample_c = -1;
    int64_t sample_frame = -1;
    int64_t sample_y = -1;
    int64_t sample_x = -1;
    float sample_actual = 0;
    float sample_expected = 0;
    uint32_t sample_actual_bits = 0;
    uint32_t sample_expected_bits = 0;
};

int fail(const std::string & message) {
    std::cerr << message << '\n';
    return 1;
}

uint32_t mix(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float unit(uint32_t & state) {
    return static_cast<float>(mix(state) >> 8) / static_cast<float>(1u << 24);
}

std::vector<float> image(int64_t height, int64_t width, uint32_t seed, bool portrait) {
    const int64_t frames = portrait ? 1 : 33;
    const size_t pixels = static_cast<size_t>(height * width);
    std::vector<float> values(static_cast<size_t>(3 * frames) * pixels);
    uint32_t state = seed;
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < 3; ++c) {
                float sample = unit(state) * 3.F - 1.5F;
                if (!portrait && frame == 0 && p == 0)
                    sample = (c == 0 ? -1.F : (c == 1 ? 0.F : 1.F));
                if (!portrait && frame == 1)
                    sample = 0.2F;
                if (!portrait && frame == 2 && p == pixels / 2)
                    sample = c == 0 ? 4.F : -4.F;
                if (!portrait && frame == 3) {
                    const float knee = 0.04045F;
                    const float encoded[] = {knee * 2.F - 1.F, (knee - 1.e-4F) * 2.F - 1.F, (knee + 1.e-4F) * 2.F - 1.F};
                    sample = encoded[c];
                }
                values[(static_cast<size_t>(c * frames + frame) * pixels) + p] = sample;
            }
        }
    }
    return values;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void note_sample(Error & error, size_t index, float actual, float expected, bool nonfinite, int64_t frames,
    int64_t height, int64_t width) {
    if (error.have_sample)
        return;
    error.have_sample = true;
    error.sample_nonfinite = nonfinite;
    error.sample_index = index;
    error.sample_actual = actual;
    error.sample_expected = expected;
    error.sample_actual_bits = float_bits(actual);
    error.sample_expected_bits = float_bits(expected);
    if (frames > 0 && height > 0 && width > 0) {
        const size_t pixels = static_cast<size_t>(height * width);
        const size_t plane = static_cast<size_t>(frames) * pixels;
        if (plane > 0 && index / plane < 3) {
            error.sample_c = static_cast<int64_t>(index / plane);
            const size_t rem = index % plane;
            error.sample_frame = static_cast<int64_t>(rem / pixels);
            const size_t p = rem % pixels;
            error.sample_y = static_cast<int64_t>(p / static_cast<size_t>(width));
            error.sample_x = static_cast<int64_t>(p % static_cast<size_t>(width));
        }
    }
}

Error compare(const std::vector<float> & actual, const std::vector<float> & expected, int64_t frames = 0,
    int64_t height = 0, int64_t width = 0) {
    if (actual.size() != expected.size())
        throw std::runtime_error("color output size mismatch actual=" + std::to_string(actual.size()) +
                                 " expected=" + std::to_string(expected.size()));
    Error error;
    error.bitwise = actual.size() == expected.size() &&
                    std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
    double square = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const bool finite_actual = std::isfinite(actual[i]);
        const bool finite_expected = std::isfinite(expected[i]);
        if (!finite_actual)
            ++error.nonfinite_actual;
        if (!finite_expected)
            ++error.nonfinite_expected;
        if (!finite_actual || !finite_expected) {
            note_sample(error, i, actual[i], expected[i], true, frames, height, width);
            continue;
        }
        ++error.finite_pairs;
        const double delta = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        const double abs_delta = std::abs(delta);
        if (actual[i] != expected[i])
            note_sample(error, i, actual[i], expected[i], false, frames, height, width);
        error.max_abs = std::max(error.max_abs, abs_delta);
        square += delta * delta;
        const auto qa = static_cast<int>(std::lround(std::clamp((actual[i] + 1.F) * 127.5F, 0.F, 255.F)));
        const auto qb = static_cast<int>(std::lround(std::clamp((expected[i] + 1.F) * 127.5F, 0.F, 255.F)));
        error.rgb8_mae += std::abs(qa - qb);
        if (qa != qb)
            ++error.rgb8_mismatch;
    }
    if (error.finite_pairs > 0) {
        error.rmse = std::sqrt(square / static_cast<double>(error.finite_pairs));
        error.rgb8_mae /= static_cast<double>(error.finite_pairs);
    }
    return error;
}

void print_error(const std::string & name, const Error & error) {
    std::cout << name << " max_abs=" << error.max_abs << " rmse=" << error.rmse
              << " finite_pairs=" << error.finite_pairs << " nonfinite_actual=" << error.nonfinite_actual
              << " nonfinite_expected=" << error.nonfinite_expected << " rgb8_mismatch=" << error.rgb8_mismatch
              << " rgb8_mae=" << error.rgb8_mae << " bitwise=" << (error.bitwise ? 1 : 0);
    if (error.have_sample) {
        std::cout << " first_index=" << error.sample_index << " c=" << error.sample_c
                  << " frame=" << error.sample_frame << " y=" << error.sample_y << " x=" << error.sample_x
                  << " actual=" << error.sample_actual << " expected=" << error.sample_expected << std::hex
                  << " actual_bits=0x" << error.sample_actual_bits << " expected_bits=0x" << error.sample_expected_bits
                  << std::dec << " sample_nonfinite=" << (error.sample_nonfinite ? 1 : 0);
    }
    std::cout << '\n';
}

void require_close(const std::string & name, const Error & error) {
    print_error(name, error);
    if (error.nonfinite_actual != 0 || error.nonfinite_expected != 0)
        throw std::runtime_error(name + " has non-finite samples actual=" + std::to_string(error.nonfinite_actual) +
                                 " expected=" + std::to_string(error.nonfinite_expected) +
                                 " max_abs=" + std::to_string(error.max_abs));
    if (error.max_abs > kMaxAbsGate || error.rmse > kRmseGate)
        throw std::runtime_error(name + " exceeded the numerical gate max_abs=" + std::to_string(error.max_abs) +
                                 " rmse=" + std::to_string(error.rmse) + " limit_max_abs=" +
                                 std::to_string(kMaxAbsGate) + " limit_rmse=" + std::to_string(kRmseGate));
}

std::vector<float> read_tensor(const ggml_tensor * tensor) {
    std::vector<float> values;
    engine::core::read_tensor_float_into(tensor, values);
    return values;
}

void check_shape(engine::core::ExecutionContext & execution, engine::models::soulx_flashhead::ColorMatcher & graph,
    engine::models::soulx_flashhead::ColorMatcher & fused, int64_t height, int64_t width, float strength) {
    namespace fh = engine::models::soulx_flashhead;
    const auto video = image(height, width, 7 + static_cast<uint32_t>(height * 17 + width), false);
    const auto portrait = image(height, width, 99 + static_cast<uint32_t>(width), true);
    const std::string name = std::to_string(height) + "x" + std::to_string(width) + " strength=" + std::to_string(strength);
    const auto & graph_host = graph.match(video, portrait, height, width, strength);
    const auto & fused_host = fused.match(video, portrait, height, width, strength);
    const auto & fused_repeat = fused.match(video, portrait, height, width, strength);
    if (fused_repeat.size() != fused_host.size() ||
        std::memcmp(fused_repeat.data(), fused_host.data(), fused_host.size() * sizeof(float)) != 0)
        throw std::runtime_error(name + " fused repeat was not bitwise identical");
    if (strength == 0.F) {
        if (&graph_host != &video || &fused_host != &video)
            throw std::runtime_error(name + " strength 0 must borrow the input");
    } else {
        require_close(name + " host", compare(fused_host, graph_host));
    }
    engine::core::ConstantTensorCache cache(execution.backend(), execution.config().threads, "color.opt." + name,
        8 * 1024 * 1024);
    const auto device_video = cache.make_f32(engine::core::TensorShape::from_dims({3, 33, height, width}), video);
    cache.ensure_uploaded();
    for (const int tail : {9, 5}) {
        const auto graph_device = graph.match_device(device_video, portrait, height, width, strength, tail);
        const auto fused_device = fused.match_device(device_video, portrait, height, width, strength, tail);
        const auto graph_video = read_tensor(graph_device.video.tensor);
        const auto fused_video = read_tensor(fused_device.video.tensor);
        const auto graph_motion = read_tensor(graph_device.motion.tensor);
        const auto fused_motion = read_tensor(fused_device.motion.tensor);
        const size_t frames = static_cast<size_t>(33 - tail);
        const size_t pixels = static_cast<size_t>(height * width);
        if (graph_video.size() != 3 * frames * pixels || graph_motion.size() != 3 * static_cast<size_t>(tail) * pixels)
            throw std::runtime_error(name + " unexpected device shape");
        if (strength == 0.F) {
            if (fused_video != graph_video || fused_motion != graph_motion)
                throw std::runtime_error(name + " strength 0 device copy was not exact");
        } else {
            require_close(name + " device tail=" + std::to_string(tail), compare(fused_video, graph_video));
            require_close(name + " motion tail=" + std::to_string(tail), compare(fused_motion, graph_motion));
        }
        const auto & host_again = fused.match(video, portrait, height, width, strength);
        if (strength != 0.F) {
            // Host match() always runs the CPU double/libm kernel. Device match on
            // Metal runs the shader. Those are not the same bits. CPU device and
            // host are the same kernel and stay bitwise.
            if (host_again.size() != 3 * 33 * pixels)
                throw std::runtime_error(name + " fused host size " + std::to_string(host_again.size()));
            std::vector<float> host_window(fused_video.size());
            for (size_t channel = 0; channel < 3; ++channel)
                std::memcpy(host_window.data() + channel * frames * pixels,
                    host_again.data() + (channel * 33 + static_cast<size_t>(tail)) * pixels,
                    frames * pixels * sizeof(float));
            const char * device_name = execution.backend_type() == engine::core::BackendType::Metal ? "metal-shader"
                                                                                                    : "cpu-double";
            const auto window_error = compare(fused_video, host_window, static_cast<int64_t>(frames), height, width);
            const std::string window_name = name + " tail=" + std::to_string(tail) +
                                            " fused-device(" + device_name + ") vs fused-host(cpu-double-libm)";
            require_close(window_name, window_error);
            if (execution.backend_type() != engine::core::BackendType::Metal && !window_error.bitwise)
                throw std::runtime_error(window_name + " CPU device and host differ bitwise max_abs=" +
                                         std::to_string(window_error.max_abs));
        }
    }
}

void check_threads(engine::core::ExecutionContext & execution) {
    namespace fh = engine::models::soulx_flashhead;
    fh::ColorMatcherConfig one;
    one.impl = fh::ColorCorrectionImpl::Fused;
    one.threads = 1;
    fh::ColorMatcherConfig two = one;
    two.threads = 2;
    fh::ColorMatcher a(execution, one);
    fh::ColorMatcher b(execution, two);
    const auto video = image(8, 6, 1234, false);
    const auto portrait = image(8, 6, 4321, true);
    const auto & left = a.match(video, portrait, 8, 6, 0.5F);
    const auto & right = b.match(video, portrait, 8, 6, 0.5F);
    if (left.size() != right.size() || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) != 0)
        throw std::runtime_error("fused thread count changed the result");
    std::cout << "threads 1 vs 2 exact\n";
}

void check_portrait_sensitivity(engine::core::ExecutionContext & execution) {
    namespace fh = engine::models::soulx_flashhead;
    fh::ColorMatcherConfig config;
    config.impl = fh::ColorCorrectionImpl::Fused;
    config.threads = 1;
    fh::ColorMatcher fused(execution, config);
    const auto video = image(4, 4, 11, false);
    auto portrait = image(4, 4, 13, true);
    const auto first = fused.match(video, portrait, 4, 4, 1.F);
    const std::vector<float> saved = first;
    portrait[0] += 0.25F;
    const auto second = fused.match(video, portrait, 4, 4, 1.F);
    if (saved.size() != second.size() || std::memcmp(saved.data(), second.data(), saved.size() * sizeof(float)) == 0)
        throw std::runtime_error("changing the portrait did not change fused color");
    std::cout << "portrait cache invalidates\n";
}

void check_motion_rejection(engine::models::soulx_flashhead::ColorMatcher & matcher,
    engine::core::ExecutionContext & execution) {
    const auto video = image(2, 2, 3, false);
    const auto portrait = image(2, 2, 5, true);
    engine::core::ConstantTensorCache cache(execution.backend(), 1, "color.opt.reject", 1024 * 1024);
    const auto device = cache.make_f32(engine::core::TensorShape::from_dims({3, 33, 2, 2}), video);
    cache.ensure_uploaded();
    bool threw = false;
    try {
        matcher.match_device(device, portrait, 2, 2, 1.F, 17);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    if (!threw)
        throw std::runtime_error("oversized motion tail was accepted");
}

void check_metal_float_row(const std::vector<float> & row, const char * name) {
    const auto metal = engine::models::soulx_flashhead::color_fused_metal_f32_row(row.data(),
        static_cast<int64_t>(row.size()));
    double sum = 0;
    for (float value : row)
        sum += static_cast<double>(value);
    const float mean = static_cast<float>(sum) / static_cast<float>(row.size());
    double squares = 0;
    for (float value : row) {
        const float delta = value - mean;
        squares += static_cast<double>(delta * delta);
    }
    const float stddev = std::sqrt(static_cast<float>(squares) / static_cast<float>(row.size()));
    const double mean_abs = std::abs(static_cast<double>(metal.mean) - mean);
    const double std_abs = std::abs(static_cast<double>(metal.stddev) - static_cast<double>(stddev));
    std::cout << "metal-f32 row " << name << " n=" << row.size() << " mean_abs=" << mean_abs
              << " std_abs=" << std_abs << " metal_std=" << metal.stddev << " double_std=" << stddev << '\n';
    if (!std::isfinite(metal.mean) || !std::isfinite(metal.stddev))
        throw std::runtime_error(std::string(name) + " metal float stats were non-finite");
    if (mean_abs > 1e-4 || std_abs > 1e-4)
        throw std::runtime_error(std::string(name) + " metal float row stats diverged from the double sum");
}

bool has_identifier(const std::string & source, const std::string & word) {
    std::size_t pos = 0;
    while ((pos = source.find(word, pos)) != std::string::npos) {
        const bool left = pos == 0 ||
                          (std::isalnum(static_cast<unsigned char>(source[pos - 1])) == 0 && source[pos - 1] != '_');
        const std::size_t end = pos + word.size();
        const bool right = end >= source.size() ||
                           (std::isalnum(static_cast<unsigned char>(source[end])) == 0 && source[end] != '_');
        if (left && right)
            return true;
        pos = end;
    }
    return false;
}

void check_metal_shader_source() {
    namespace fh = engine::models::soulx_flashhead;
    if (fh::kMetalStatsLanes != 256)
        throw std::runtime_error("Metal stats lane count drifted from the shader");
    const std::string source(fh::kFlashHeadColorMetalShader);
    if (source.find("#pragma METAL fp math_mode(safe)") == std::string::npos ||
        source.find("#pragma METAL fp contract(off)") == std::string::npos ||
        source.find("#pragma STDC FP_CONTRACT OFF") == std::string::npos)
        throw std::runtime_error("Metal shader is missing the strict floating-point pragmas");
    if (source.find("kahan_add") == std::string::npos || source.find("two_sum") == std::string::npos ||
        source.find("256u") == std::string::npos)
        throw std::runtime_error("Metal shader stats kernel does not match the 256-lane float reduction");
    if (has_identifier(source, "double") || has_identifier(source, "long"))
        throw std::runtime_error("Metal shader source contains a 64-bit scalar type");
    std::cout << "metal shader source is float-only with contract off\n";
    std::cout << "this check reads the shader text; it does not compile or run MSL\n";
}

void check_metal_float_rows() {
    check_metal_float_row(std::vector<float>(1, 0.25F), "n1");
    check_metal_float_row(std::vector<float>(35, -0.5F), "constant-35");
    check_metal_float_row(std::vector<float>(256, 0.2F), "constant-256");
    check_metal_float_row(std::vector<float>(257, 0.2F), "constant-257");
    check_metal_float_row(std::vector<float>(147456, 0.2F), "constant-384sq");
    check_metal_float_row(std::vector<float>(262144, -1.25F), "constant-512sq");
    std::vector<float> spike(262144, 0.F);
    spike[1000] = 4.F;
    spike[262143] = -3.F;
    check_metal_float_row(spike, "spike-512sq");
    std::vector<float> random(147456);
    uint32_t state = 99;
    for (float & value : random)
        value = unit(state) * 80.F - 40.F;
    check_metal_float_row(random, "random-lab-384sq");
}

void check_metal_float_color(engine::models::soulx_flashhead::ColorMatcher & graph, int64_t height, int64_t width,
    float strength, uint32_t seed, const char * label) {
    namespace fh = engine::models::soulx_flashhead;
    const auto video = image(height, width, seed, false);
    const auto portrait = image(height, width, seed + 17, true);
    const auto & graph_host = graph.match(video, portrait, height, width, strength);
    std::vector<float> fused(video.size());
    fh::ColorReferenceCache cache;
    fh::ColorFusedTiming timing;
    fh::color_fused_apply_cpu(video.data(), portrait.data(), height, width, 0, 33, 0, strength, 2, cache, fused.data(),
        nullptr, timing, fh::ColorFusedReduction::MetalFloat);
    const auto error = compare(fused, graph_host, 33, height, width);
    require_close(std::string("metal-f32-replica ") + label + " " + std::to_string(height) + "x" +
                      std::to_string(width) + " cpu-libm-float-stats vs graph",
        error);
}

void check_exec() {
    namespace fh = engine::models::soulx_flashhead;
    const auto base = fh::resolve_flashhead_exec({}, engine::core::BackendType::Cpu, true);
    if (base.vae_cudnn || base.color != fh::ColorCorrectionImpl::Graph)
        throw std::runtime_error("default exec plan changed");
    const auto fused = fh::resolve_flashhead_exec(
        {{"soulx_flashhead.exec_plan", "{\"vae\":{\"impl\":\"ggml\"},\"color\":{\"impl\":\"fused\"}}"}},
        engine::core::BackendType::Cpu, true);
    if (fused.vae_cudnn || fused.color != fh::ColorCorrectionImpl::Fused)
        throw std::runtime_error("fused exec plan was not selected");
    const auto cudnn = fh::resolve_flashhead_exec(
        {{"soulx_flashhead.exec_plan", "{\"vae\":{\"impl\":\"cudnn\"}}"}}, engine::core::BackendType::Cuda, true);
    if (!cudnn.vae_cudnn || cudnn.color != fh::ColorCorrectionImpl::Graph)
        throw std::runtime_error("cudnn plan disturbed the default color path");
    auto expect_throw = [](auto && call, const char * label) {
        bool threw = false;
        try {
            call();
        } catch (const std::runtime_error &) {
            threw = true;
        }
        if (!threw)
            throw std::runtime_error(std::string(label) + " was accepted");
    };
    expect_throw(
        [] {
            fh::resolve_flashhead_exec({{"soulx_flashhead.exec_plan", "{\"color\":{\"impl\":\"fused\"}}"}},
                engine::core::BackendType::Cuda, true);
        },
        "fused on cuda");
    expect_throw(
        [] {
            fh::resolve_flashhead_exec({{"soulx_flashhead.exec_plan", "{\"vae\":{\"impl\":\"cudnn\"}}"}},
                engine::core::BackendType::Cpu, true);
        },
        "cudnn on cpu");
    expect_throw(
        [] {
            fh::resolve_flashhead_exec({{"soulx_flashhead.exec_plan", "{\"vae\":{\"impl\":\"cudnn\"}}"}},
                engine::core::BackendType::Cuda, false);
        },
        "cudnn on a non-lite variant");
    expect_throw(
        [] {
            fh::resolve_flashhead_exec({{"soulx_flashhead.exec_plan", "{\"color\":{\"impl\":\"approx\"}}"}},
                engine::core::BackendType::Cpu, true);
        },
        "unknown color impl");
    if (!fh::color_fused_metal_compiled()) {
        expect_throw(
            [] {
                fh::resolve_flashhead_exec({{"soulx_flashhead.exec_plan", "{\"color\":{\"impl\":\"fused\"}}"}},
                    engine::core::BackendType::Metal, true);
            },
            "fused metal without the shader");
    }
    std::cout << "exec plan gates ok, metal shader compiled=" << (fh::color_fused_metal_compiled() ? "yes" : "no")
              << '\n';
}

void bench(engine::core::ExecutionContext & execution, int64_t side) {
    namespace fh = engine::models::soulx_flashhead;
    fh::ColorMatcher graph(execution);
    fh::ColorMatcherConfig config;
    config.impl = fh::ColorCorrectionImpl::Fused;
    config.threads = execution.config().threads;
    fh::ColorMatcher fused(execution, config);
    const auto video = image(side, side, 42, false);
    const auto portrait = image(side, side, 43, true);
    engine::core::ConstantTensorCache cache(execution.backend(), execution.config().threads, "color.opt.bench",
        64 * 1024 * 1024);
    const auto device = cache.make_f32(engine::core::TensorShape::from_dims({3, 33, side, side}), video);
    cache.ensure_uploaded();
    graph.match_device(device, portrait, side, side, 1.F, 9);
    fused.match_device(device, portrait, side, side, 1.F, 9);
    const auto graph_started = std::chrono::steady_clock::now();
    const auto graph_out = graph.match_device(device, portrait, side, side, 1.F, 9);
    const double graph_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - graph_started).count();
    const auto fused_started = std::chrono::steady_clock::now();
    const auto fused_out = fused.match_device(device, portrait, side, side, 1.F, 9);
    const double fused_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fused_started).count();
    const auto error = compare(read_tensor(fused_out.video.tensor), read_tensor(graph_out.video.tensor));
    const char * backend_name = execution.backend_type() == engine::core::BackendType::Metal ? "metal" : "cpu";
    std::cout << "bench " << side << " backend=" << backend_name
              << " graph_ms=" << graph_ms << " fused_ms=" << fused_ms << " max_abs=" << error.max_abs
              << " rmse=" << error.rmse << " rgb8_mismatch=" << error.rgb8_mismatch << '\n';
    std::cout << "bench timings are for this process only; they are not an M5 Ultra measurement\n";
    require_close("bench " + std::to_string(side), error);
    const auto video_copy = video;
    std::vector<float> metal_float(video.size());
    fh::ColorReferenceCache reference;
    fh::ColorFusedTiming timing;
    const auto metal_started = std::chrono::steady_clock::now();
    fh::color_fused_apply_cpu(video_copy.data(), portrait.data(), side, side, 0, 33, 0, 1.F, execution.config().threads,
        reference, metal_float.data(), nullptr, timing, fh::ColorFusedReduction::MetalFloat);
    const double metal_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metal_started).count();
    const auto metal_error = compare(metal_float, graph.match(video_copy, portrait, side, side, 1.F), 33, side, side);
    std::cout << "bench metal-f32-stats " << side << " cpu_replica_ms=" << metal_ms
              << " this is the CPU replica of the shader reduction, not a GPU dispatch\n";
    require_close("bench metal-f32-replica " + std::to_string(side), metal_error);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        engine::core::BackendType backend = engine::core::BackendType::Cpu;
        int64_t bench_side = 0;
        std::string log_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--backend" && i + 1 < argc) {
                const std::string name = argv[++i];
                if (name == "cpu")
                    backend = engine::core::BackendType::Cpu;
                else if (name == "metal")
                    backend = engine::core::BackendType::Metal;
                else
                    return fail("unsupported benchmark backend: " + name);
            } else if (arg == "--bench" && i + 1 < argc) {
                bench_side = std::stoll(argv[++i]);
            } else if (arg == "--log" && i + 1 < argc) {
                log_path = argv[++i];
            } else {
                return fail("Usage: soulx_flashhead_color_opt_test [--backend cpu|metal] [--bench SIZE] [--log FILE]");
            }
        }
        if (!log_path.empty())
            engine::debug::configure_logging({true, log_path});
        check_exec();
        check_metal_shader_source();
        check_metal_float_rows();
        engine::core::BackendConfig config;
        config.type = backend;
        config.threads = 2;
        engine::core::ExecutionContext execution(config);
        if (execution.backend_type() != backend)
            return fail("requested backend was not initialized");
        engine::core::set_backend_round_bf16_fusion(execution.backend(), true);
        namespace fh = engine::models::soulx_flashhead;
        fh::ColorMatcher graph(execution);
        fh::ColorMatcherConfig fused_config;
        fused_config.impl = fh::ColorCorrectionImpl::Fused;
        fused_config.threads = 2;
        fh::ColorMatcher fused(execution, fused_config);
        check_motion_rejection(graph, execution);
        check_motion_rejection(fused, execution);
        check_threads(execution);
        check_portrait_sensitivity(execution);
        for (const float strength : {0.37F, 1.F}) {
            check_metal_float_color(graph, 1, 1, strength, 3, "edge");
            check_metal_float_color(graph, 5, 7, strength, 4, "edge");
            check_metal_float_color(graph, 8, 8, strength, 5, "edge");
            check_metal_float_color(graph, 16, 12, strength, 6, "edge");
        }
        {
            const int64_t height = 8, width = 8;
            std::vector<float> video(static_cast<size_t>(3 * 33 * height * width), 0.2F);
            std::vector<float> portrait(static_cast<size_t>(3 * height * width), -0.15F);
            const auto & graph_host = graph.match(video, portrait, height, width, 1.F);
            std::vector<float> fused(video.size());
            fh::ColorReferenceCache cache;
            fh::ColorFusedTiming timing;
            fh::color_fused_apply_cpu(video.data(), portrait.data(), height, width, 0, 33, 0, 1.F, 1, cache,
                fused.data(), nullptr, timing, fh::ColorFusedReduction::MetalFloat);
            const auto error = compare(fused, graph_host, 33, height, width);
            require_close("metal-f32-replica constant 8x8 cpu-libm-float-stats vs graph", error);
        }
        for (const float strength : {0.F, 0.37F, 1.F}) {
            check_shape(execution, graph, fused, 1, 1, strength);
            check_shape(execution, graph, fused, 5, 7, strength);
            check_shape(execution, graph, fused, 8, 8, strength);
            check_shape(execution, graph, fused, 16, 12, strength);
        }
        if (bench_side > 0) {
            if (bench_side > 512)
                return fail("bench size is limited to 512");
            bench(execution, bench_side);
        }
        std::cout << "soulx_flashhead_color_opt_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        return fail(error.what());
    }
}
