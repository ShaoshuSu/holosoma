#include "holosoma_cpp/inference_engine.hpp"

#include <cmath>
#include <stdexcept>

#include "holosoma_cpp/locomotion_policy.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace holosoma {
namespace {
class MemoryRobot final : public RobotInterface {
 public:
  RobotState state;
  JointCommand command;
  bool have_state = false;
  bool read_state(RobotState& out) override {
    out = state;
    return have_state;
  }
  void send_command(const JointCommand& out) override { command = out; }
  std::optional<WirelessController> read_wireless_controller() override { return std::nullopt; }
  std::string description() const override { return "in-process observations"; }
};
class MemoryVelocity final : public VelocityProvider {
 public:
  std::optional<VelCmd> poll_velocity() override { return std::nullopt; }
  void zero() override {}
};
}  // namespace

struct InferenceEngine::Impl {
  MemoryRobot robot;
  MemoryVelocity velocity;
  PolicyContext context;
  std::unique_ptr<BasePolicy> policy;
  bool autostart = false;
  bool pending_engagement = false;
};

InferenceEngine::InferenceEngine(InferenceConfig config, bool autostart) : impl_(std::make_unique<Impl>()) {
  validate_inference_config(config);
  impl_->context.robot = &impl_->robot;
  impl_->context.velocity_input = &impl_->velocity;
  if (config.is_wbt())
    impl_->policy = std::make_unique<WholeBodyTrackingPolicy>(config, impl_->context);
  else
    impl_->policy = std::make_unique<LocomotionPolicy>(config, impl_->context);
  impl_->policy->initialize();
  impl_->autostart = autostart;
  impl_->pending_engagement = autostart;
  if (autostart) {
    impl_->policy->force_policy_active();
    if (config.is_wbt()) impl_->policy->dispatch_command(StateCommand::START_MOTION_CLIP);
  }
}
InferenceEngine::~InferenceEngine() = default;

std::vector<double> InferenceEngine::step(const std::vector<float>& observation) {
  auto offsets = impl_->policy->infer_observation(observation);
  const auto& cfg = impl_->policy->config().robot;
  if (offsets.size() > cfg.default_dof_angles.size()) throw std::runtime_error("Action exceeds robot dimension");
  offsets.insert(offsets.begin(), cfg.default_dof_angles.size() - offsets.size(), 0.0);
  for (size_t i = 0; i < offsets.size(); ++i) {
    offsets[i] += cfg.default_dof_angles[i];
    if (cfg.joint_offsets_deg) offsets[i] += (*cfg.joint_offsets_deg)[i] * 3.14159265358979323846 / 180.0;
    if (!std::isfinite(offsets[i])) throw std::runtime_error("Non-finite joint target");
  }
  return offsets;
}

void InferenceEngine::set_state(const RobotState& state) {
  const auto n = static_cast<size_t>(impl_->policy->config().robot.num_joints);
  if (state.base_pos.size() != 3 || state.base_quat_wxyz.size() != 4 || state.base_lin_vel.size() != 3 ||
      state.base_ang_vel.size() != 3 || state.dof_pos.size() != n || state.dof_vel.size() != n)
    throw std::invalid_argument("Robot state dimensions do not match the configuration");
  for (const auto* values : {&state.base_pos, &state.base_quat_wxyz, &state.base_lin_vel, &state.base_ang_vel,
                             &state.dof_pos, &state.dof_vel})
    for (double v : *values)
      if (!std::isfinite(v)) throw std::invalid_argument("Robot state must be finite");
  double norm = 0;
  for (double v : state.base_quat_wxyz) norm += v * v;
  if (norm < 1e-16) throw std::invalid_argument("Robot orientation must be nonzero");
  if (state.projected_gravity) {
    if (state.projected_gravity->size() != 3)
      throw std::invalid_argument("Corrected gravity must have three components");
    for (double v : *state.projected_gravity)
      if (!std::isfinite(v)) throw std::invalid_argument("Corrected gravity must be finite");
  }
  impl_->robot.state = state;
  impl_->robot.have_state = true;
}
JointCommand InferenceEngine::step_state(const RobotState& state) {
  set_state(state);
  if (impl_->pending_engagement) {
    impl_->policy->handle_start_policy();
    if (impl_->policy->config().is_wbt()) impl_->policy->dispatch_command(StateCommand::START_MOTION_CLIP);
    impl_->pending_engagement = false;
  }
  impl_->policy->policy_action(state);
  return impl_->robot.command;
}
void InferenceEngine::command(StateCommand cmd) {
  if (cmd == StateCommand::START || cmd == StateCommand::STOP || cmd == StateCommand::INIT)
    impl_->pending_engagement = false;
  impl_->policy->dispatch_command(cmd);
}
void InferenceEngine::velocity(const VelCmd& cmd) { impl_->policy->apply_velocity(cmd); }
void InferenceEngine::update_phase() {
  if (impl_->policy->use_phase()) impl_->policy->update_phase_time();
}
void InferenceEngine::reset() {
  impl_->policy->reset();
  impl_->pending_engagement = impl_->autostart;
  if (impl_->autostart) {
    impl_->policy->force_policy_active();
    if (impl_->policy->config().is_wbt()) impl_->policy->dispatch_command(StateCommand::START_MOTION_CLIP);
  }
}
const BasePolicy& InferenceEngine::policy() const { return *impl_->policy; }
}  // namespace holosoma
