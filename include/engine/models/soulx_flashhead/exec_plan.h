#pragma once

#include "engine/framework/core/module.h"
#include "engine/models/soulx_flashhead/color.h"

#include <string>
#include <unordered_map>

namespace engine::models::soulx_flashhead {

struct FlashHeadExecSelection {
    bool vae_cudnn = false;
    ColorCorrectionImpl color = ColorCorrectionImpl::Graph;
};

// `backend` is the session backend. `lite_variant` is required for vae.impl=cudnn.
// color.impl=graph is the default. color.impl=fused keeps correction enabled and
// is rejected unless the session backend is CPU or Metal.
FlashHeadExecSelection resolve_flashhead_exec(
    const std::unordered_map<std::string, std::string> & options,
    core::BackendType backend,
    bool lite_variant);

} // namespace engine::models::soulx_flashhead
