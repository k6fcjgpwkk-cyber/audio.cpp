#pragma once
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include <memory>
#include <vector>
namespace engine::models::soulx_flashhead {
struct ColorMatchResult {
    core::TensorValue video;
    core::TensorValue motion;
};

// Graph is the default and is unchanged. Fused is opt-in: same Lab transfer and
// the same bf16-rounded constants, evaluated in fewer passes. It is not a
// promise of bit-identical Metal results.
enum class ColorCorrectionImpl {
    Graph = 0,
    Fused = 1,
};

// True only when this binary was built with the Metal fused kernel.
bool color_fused_metal_compiled();

struct ColorMatcherConfig {
    ColorCorrectionImpl impl = ColorCorrectionImpl::Graph;
    // Positive values override the execution context. Negative uses that context.
    int threads = -1;
};

class ColorMatcher {
  public:
    explicit ColorMatcher(core::ExecutionContext & execution);
    ColorMatcher(core::ExecutionContext & execution, ColorMatcherConfig config);
    ~ColorMatcher();
    // Borrows the input when disabled; otherwise the cached graph or fused output.
    const std::vector<float> & match(const std::vector<float> & video, const std::vector<float> & portrait,
                                     int64_t height, int64_t width, float strength);

    // Keeps corrected video and a contiguous motion tail on the backend.
    // Valid until the next match_device for the same shape, or destruction.
    ColorMatchResult match_device(const core::TensorValue & video, const std::vector<float> & portrait,
        int64_t height, int64_t width, float strength, int64_t motion_frames);

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace engine::models::soulx_flashhead
