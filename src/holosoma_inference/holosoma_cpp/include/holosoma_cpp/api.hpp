// The public surface of holosoma_cpp. Include this and nothing else.
//
//   Policy policy(preset_path, checkpoint_path);
//   const JointCommand& command = policy.step(state);
//
// Owns the PolicyContext, the input providers a policy may have no use for, and an offline
// transport, so step() returns the command rather than publishing it. Re-exports RobotState,
// JointCommand, and OfflineInterface (a transport for shadow mode and tests).
#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "holosoma_cpp/config.hpp"
#include "holosoma_cpp/controller.hpp"
#include "holosoma_cpp/input.hpp"
#include "holosoma_cpp/policy.hpp"
#include "holosoma_cpp/robot_interface.hpp"

namespace holosoma {

// Reads nothing, publishes nowhere, keeps what it was given. Shadow mode by construction: step
// the policy with no robot attached and read back what would have been sent.
class OfflineInterface final : public RobotInterface {
 public:
  bool read_state(RobotState& out) override {
    out = state_;
    return have_state_;
  }
  void send_command(const JointCommand& command) override { command_ = command; }
  std::optional<WirelessController> read_wireless_controller() override { return std::nullopt; }
  std::string description() const override { return "offline"; }

  void set_state(const RobotState& state) {
    state_ = state;
    have_state_ = true;
  }
  const JointCommand& command() const { return command_; }

 private:
  RobotState state_;
  JointCommand command_;
  bool have_state_ = false;
};

namespace detail {

struct NoVelocity final : VelocityProvider {
  std::optional<VelCmd> poll_velocity() override { return std::nullopt; }
  void zero() override {}
};

struct NoCommands final : CommandProvider {
  std::vector<StateCommand> poll_commands() override { return {}; }
  void bind(const std::string&, StateCommand) override {}
};

}  // namespace detail

// A policy you can step. The class comes from the preset, exactly as the CLI resolves it, so
// this serves locomotion and whole-body tracking without knowing which it has.
class Policy {
 public:
  Policy(const std::string& preset, const std::string& checkpoint)
      : config_(parse_inference_config(YAML::LoadFile(preset), ".")) {
    if (!checkpoint.empty()) config_.task.model_path = {checkpoint};
    context_.robot = &robot_;
    context_.velocity_input = &velocity_;
    context_.command_provider = &commands_;
    policy_ = make_policy(config_, context_);
    policy_->initialize();
    policy_->force_policy_active();
  }

  // One tick. The command is returned rather than published.
  const JointCommand& step(const RobotState& state) {
    robot_.set_state(state);
    policy_->policy_action(state);
    return robot_.command();
  }

  int num_dofs() const { return static_cast<int>(config_.robot.dof_names.size()); }
  const std::vector<std::string>& dof_names() const { return config_.robot.dof_names; }
  const InferenceConfig& config() const { return config_; }
  BasePolicy& policy() { return *policy_; }  // escape hatch

 private:
  InferenceConfig config_;
  OfflineInterface robot_;
  detail::NoVelocity velocity_;
  detail::NoCommands commands_;
  PolicyContext context_;
  std::unique_ptr<BasePolicy> policy_;  // declared after the members it borrows
};

}  // namespace holosoma
