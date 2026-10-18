// Actor-only whole-body tracking: the reference comes from the host, not from the graph.
//
// The embedded-motion WBT policy runs a graph emitting joint_pos/joint_vel/ref_quat_xyzw
// alongside actions and feeds its own reference back. A 0p2s actor-only export
// (wbt_onnx_export_mode = actor_only) emits `actions` alone, takes (obs, time_step), and is
// driven by a window the caller supplies.
//
// The observation layout is config, not code: two groups, actor_obs at history 9 and
// actor_targets at history 1, terms sorted by their zero-padded names. The only assembly added
// here is concatenating the two groups, which is what the export's flat input expects.
#pragma once

#include <array>
#include <set>
#include <string>
#include <vector>

#include "holosoma_cpp/policy.hpp"

namespace holosoma {

constexpr int kNumOffsets = 5;  // 0p2s: frames 0, 2, 4, 6, 8

// The five future offsets of the reference, in the export's dof order. Host-supplied.
struct ReferenceWindow {
  std::vector<double> q;                    // kNumOffsets * num_dofs, offset-major
  std::vector<double> dq;                   // kNumOffsets * num_dofs
  std::vector<std::array<double, 4>> quat;  // kNumOffsets, wxyz
};

class GeneralWBTPolicy : public BasePolicy {
 public:
  GeneralWBTPolicy(InferenceConfig config, PolicyContext& context);

  // Set before each step(); borrowed, never owned, so a clip and a live stream are swappable.
  void set_reference(const ReferenceWindow* window) { reference_ = window; }

 protected:
  void setup_policy(ModelSlot& slot) override;
  std::set<std::string> available_terms() const override;
  void compute_terms(const RobotState& state, TermValues& terms) override;
  std::vector<double> rl_inference(const RobotState& state) override;

 private:
  const std::vector<float>& prepare_flat_obs(const RobotState& state);

  const ReferenceWindow* reference_ = nullptr;
  std::vector<double> policy_default_pos_;
  std::vector<double> action_scale_;
  std::array<double, 2> yaw0_{};
  bool yaw0_captured_ = false;
  std::vector<float> flat_obs_;
};

}  // namespace holosoma
