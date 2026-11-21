#include "engine/models/soulx_flashhead/exec_plan.h"

#include "color_fused.h"
#include "engine/framework/runtime/options.h"

#include <stdexcept>

namespace engine::models::soulx_flashhead {

FlashHeadExecSelection resolve_flashhead_exec(const std::unordered_map<std::string, std::string> & options,
    core::BackendType backend, bool lite_variant) {
    const auto plan = runtime::parse_exec_plan(options, "soulx_flashhead.exec_plan",
        {{"vae", "ggml", {{"ggml", {backend}, {}, {}, {}}, {"cudnn", {core::BackendType::Cuda}, {}, {}, {}}}},
            {"color", "graph",
                {{"graph", {backend}, {}, {}, {}},
                    {"fused", {core::BackendType::Cpu, core::BackendType::Metal}, {}, {}, {}}}}});
    FlashHeadExecSelection selected;
    selected.vae_cudnn = plan.at("vae").impl == "cudnn";
    if (selected.vae_cudnn && backend != core::BackendType::Cuda)
        throw std::runtime_error("soulx_flashhead.exec_plan.vae: cudnn requires the CUDA session backend");
    if (selected.vae_cudnn && !lite_variant)
        throw std::runtime_error("soulx_flashhead.exec_plan.vae: cudnn requires the Lite LTX VAE");
    const auto & color = plan.at("color");
    if (color.impl == "fused") {
        selected.color = ColorCorrectionImpl::Fused;
        if (backend != core::BackendType::Cpu && backend != core::BackendType::Metal)
            throw std::runtime_error(
                "soulx_flashhead.exec_plan.color: fused supports the cpu and metal backends only");
        if (backend == core::BackendType::Metal && !color_fused_metal_compiled())
            throw std::runtime_error(
                "soulx_flashhead.exec_plan.color: fused Metal kernel is not compiled into this binary; "
                "rebuild on macOS with ENGINE_ENABLE_METAL=ON");
    }
    return selected;
}

} // namespace engine::models::soulx_flashhead
