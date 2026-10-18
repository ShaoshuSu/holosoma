#include "holosoma_cpp/general_wbt_policy.hpp"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "holosoma_cpp/math.hpp"

namespace holosoma {
namespace {

constexpr double kActionClip = 20.0;  // the export's configured action clip
constexpr const char* kObsInput = "obs";
constexpr const char* kTimeStepInput = "time_step";
constexpr const char* kActionsOutput = "actions";

// Zero because this checkpoint was trained that way: its experiment_config carries
// enable_smpl_encoder = false, and teleop is gated on the SMPL path. obs15_encoder_index's
// [1, 0, 0] one-hot selects the live modality. A SMPL-enabled export would need these
// produced -- so these zeros belong in config read from the checkpoint, not in code.
const std::vector<std::pair<const char*, int>> kInactive{
    {"obs08_teleop_lower_body_future", 240},        {"obs09_teleop_tracker_positions", 9},
    {"obs10_teleop_tracker_quaternions", 12},       {"obs11_teleop_anchor_rotation", 6},
    {"obs12_smpl_joints_local_future", 720},        {"obs13_smpl_root_rotation_future", 60},
    {"obs14_smpl_wrist_joint_positions_future", 60}};

std::vector<double> number_list(const std::string& raw, int expected) {
  const auto json = nlohmann::json::parse(raw);
  std::vector<double> out;
  if (json.is_number()) {
    out.assign(static_cast<size_t>(expected), json.get<double>());
  } else {
    for (const auto& v : json) out.push_back(v.get<double>());
  }
  if (static_cast<int>(out.size()) != expected) throw std::runtime_error("metadata width mismatch");
  return out;
}

// math.hpp has remove_yaw_offset / subtract_frame_transforms / matrix_from_quat but not these.
math::Mat3 multiply(const math::Mat3& a, const math::Mat3& b) {
  math::Mat3 out{};
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      for (int k = 0; k < 3; ++k)
        out[static_cast<size_t>(row * 3 + col)] +=
            a[static_cast<size_t>(row * 3 + k)] * b[static_cast<size_t>(k * 3 + col)];
  return out;
}

math::Mat3 transpose(const math::Mat3& r) { return {r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]}; }

double heading(const math::Mat3& r) { return std::atan2(r[3], r[0]); }

math::Mat3 strip_yaw(const math::Mat3& r, double yaw) {
  const double c = std::cos(-yaw), s = std::sin(-yaw);
  return multiply(math::Mat3{c, -s, 0, s, c, 0, 0, 0, 1}, r);
}

}  // namespace

GeneralWBTPolicy::GeneralWBTPolicy(InferenceConfig config, PolicyContext& context)
    : BasePolicy(std::move(config), context) {
  kind_ = "GeneralWBTPolicy";
}

void GeneralWBTPolicy::setup_policy(ModelSlot& slot) {
  slot.model = std::make_unique<OnnxModel>(slot.path);
  const OnnxModel& model = *slot.model;
  const auto mode = model.metadata_value("wbt_onnx_export_mode").value_or("");
  if (mode != "actor_only" && mode != "\"actor_only\"") {
    throw std::runtime_error(slot.path + ": expected an actor_only export");
  }
  if (model.find_input(kObsInput) == nullptr || model.find_input(kTimeStepInput) == nullptr) {
    throw std::runtime_error(slot.path + ": expected inputs (obs, time_step)");
  }
  policy_default_pos_ = number_list(*model.metadata_value("default_joint_pos"), num_dofs_);
  action_scale_ = number_list(*model.metadata_value("action_scale"), num_dofs_);
  slot.kp = number_list(*model.metadata_value("kp"), config_.robot.num_motors);
  slot.kd = number_list(*model.metadata_value("kd"), config_.robot.num_motors);
  slot.gain_source = "ONNX metadata";
  yaw0_captured_ = false;
}

std::set<std::string> GeneralWBTPolicy::available_terms() const {
  std::set<std::string> terms{"obs0_projected_gravity",
                              "obs1_base_ang_vel",
                              "obs2_dof_pos",
                              "obs3_dof_vel",
                              "obs4_actions",
                              "obs05_motion_joint_pos_future",
                              "obs06_motion_joint_vel_future",
                              "obs07_motion_future_ref_ori_b",
                              "obs15_encoder_index"};
  for (const auto& [name, dim] : kInactive) terms.insert(name);
  return terms;
}

void GeneralWBTPolicy::compute_terms(const RobotState& state, TermValues& terms) {
  const math::Quat quat{state.base_quat_wxyz[0], state.base_quat_wxyz[1], state.base_quat_wxyz[2],
                        state.base_quat_wxyz[3]};
  const math::Vec3 g = math::quat_rotate_inverse(quat, {0.0, 0.0, -1.0});
  terms["obs0_projected_gravity"] = {g[0], g[1], g[2]};
  terms["obs1_base_ang_vel"] = state.base_ang_vel;
  auto& dof_pos = terms["obs2_dof_pos"];
  dof_pos.resize(static_cast<size_t>(num_dofs_));
  for (size_t i = 0; i < dof_pos.size(); ++i) dof_pos[i] = state.dof_pos[i] - policy_default_pos_[i];
  terms["obs3_dof_vel"] = state.dof_vel;
  terms["obs4_actions"] = last_policy_action_;

  if (reference_ == nullptr) throw std::runtime_error("set_reference() before stepping");
  terms["obs05_motion_joint_pos_future"] = reference_->q;
  terms["obs06_motion_joint_vel_future"] = reference_->dq;

  const math::Mat3 robot = math::matrix_from_quat(quat);
  if (!yaw0_captured_) {
    yaw0_ = {heading(robot), heading(math::matrix_from_quat(reference_->quat.front()))};
    yaw0_captured_ = true;
  }
  const math::Mat3 robot_stripped = strip_yaw(robot, yaw0_[0]);
  auto& rot6d = terms["obs07_motion_future_ref_ori_b"];
  rot6d.clear();
  for (const auto& q : reference_->quat) {
    const math::Mat3 rel = multiply(transpose(robot_stripped), strip_yaw(math::matrix_from_quat(q), yaw0_[1]));
    for (int i : {0, 1, 3, 4, 6, 7}) rot6d.push_back(rel[static_cast<size_t>(i)]);
  }
  terms["obs15_encoder_index"] = {1.0, 0.0, 0.0};
  for (const auto& [name, dim] : kInactive) terms[name].assign(static_cast<size_t>(dim), 0.0);
}

const std::vector<float>& GeneralWBTPolicy::prepare_flat_obs(const RobotState& state) {
  TermValues terms;
  compute_terms(state, terms);
  ObservationGroup& proprio = obs_builder_.group("actor_obs");
  ObservationGroup& targets = obs_builder_.group("actor_targets");
  flat_obs_.resize(static_cast<size_t>(proprio.dim() + targets.dim()));
  proprio.update(terms, flat_obs_.data());
  targets.update(terms, flat_obs_.data() + proprio.dim());
  // TODO: also assign last_obs_ -- print_observations() and the parity harness read it.
  return flat_obs_;
}

std::vector<double> GeneralWBTPolicy::rl_inference(const RobotState& state) {
  const std::vector<float>& obs = prepare_flat_obs(state);
  OnnxModel& model = *active_slot().model;
  std::copy(obs.begin(), obs.end(), model.input(kObsInput).begin());
  model.input(kTimeStepInput)[0] = 0.0f;
  model.run();

  const std::vector<float>& raw = model.output(kActionsOutput);
  last_policy_action_.resize(raw.size());
  scaled_policy_action_.resize(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    last_policy_action_[i] = std::clamp(static_cast<double>(raw[i]), -kActionClip, kActionClip);
    // Core adds robot.default_dof_angles; the export's own default pose is authoritative.
    scaled_policy_action_[i] =
        last_policy_action_[i] * action_scale_[i] + policy_default_pos_[i] - default_dof_angles_[i];
  }
  return scaled_policy_action_;
}

}  // namespace holosoma
