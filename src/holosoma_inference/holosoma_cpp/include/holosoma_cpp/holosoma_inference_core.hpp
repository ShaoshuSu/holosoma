#pragma once

#include "holosoma_cpp/live_types.hpp"
#include "holosoma_cpp/onnx_model.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace holosoma::live {

// Single-graph holosoma WBT policy. The graph and its complete observation
// contract are read from ONNX metadata; unsupported contracts fail at load.
class WbtPolicy {
public:
  WbtPolicy();
  ~WbtPolicy();

  WbtPolicy(const WbtPolicy &) = delete;
  WbtPolicy &operator=(const WbtPolicy &) = delete;

  bool Initialize(const std::string &model_path, bool use_gpu = false,
                  int intra_op_threads = 2);
  bool ComputeAction(const RobotObservation &obs,
                     const WbtReferenceSequence &reference, float *action_out);

  // The `time_step` graph input, in MOTION timesteps rather than policy ticks.
  // Python's TimestepUtil derives it from elapsed wall time over the control
  // interval (holosoma_inference/policies/wbt_utils.py), so a dropped or slow
  // tick must not shift the motion phase the policy is told it is at. The
  // node supplies elapsed seconds since engage; a monotonic tick counter is
  // only equivalent while every tick lands exactly on time.
  void SetElapsedSecondsFn(std::function<double()> fn) {
    elapsed_seconds_fn_ = std::move(fn);
  }
  void SetControlInterval(double interval_s);
  float LastTimeStep() const { return last_time_step_; }

  // Reanchor heading and motion phase without erasing proprio/action history
  // on pause/resume. Reset() remains the full fresh-activation reset.
  void ResetHeading();
  void Reset();

  // Rescales the model's trained lookahead offsets over a live-reference
  // horizon. A zero horizon intentionally tiles the playback cursor.
  std::vector<int> LookaheadSampleOffsets(int horizon_frames) const;
  const std::vector<int> &TrainedLookaheadOffsets() const {
    return trained_lookahead_offsets_;
  }
  int LookaheadSlots() const {
    return static_cast<int>(trained_lookahead_offsets_.size());
  }
  int TrainedLookaheadFrames() const {
    return trained_lookahead_offsets_.empty()
               ? 0
               : trained_lookahead_offsets_.back();
  }

  const std::array<float, kActionDim> &DefaultDofPos() const {
    return default_dof_pos_;
  }
  const std::array<float, kActionDim> &Kp() const { return kp_; }
  const std::array<float, kActionDim> &Kd() const { return kd_; }
  const std::vector<float> &LastObservation() const { return obs_input_; }
  bool HeadingInitialized() const { return heading_initialized_; }
  bool IsInitialized() const { return initialized_; }
  const std::string &AnchorBodyName() const { return anchor_body_name_; }

  static bool IsWbtModelFile(const std::string &model_path);

private:
  enum class TermKind {
    kMotionCommand,
    kMotionCommandLookahead,
    kMotionRefOri,
    kProjectedGravity,
    kBaseAngVel,
    kDofPos,
    kDofVel,
    kActions,
    kJointPosFuture,
    kJointVelFuture,
    kFutureRefOri,
    kZeroBlock,
    kEncoderIndex,
  };

  struct Term {
    std::string name;
    TermKind kind;
    int dim = 0;
    int history_length = 1;
    float scale = 1.0f;
    std::deque<std::vector<float>> history;
  };

  class FkBody;

  bool ParseMetadata(std::string &error);
  bool ValidateShapes(std::string &error);
  float NextTimeStep();
  bool BuildTermValue(const Term &term, const RobotObservation &obs,
                      const WbtReferenceSequence &reference,
                      const math::Quatf &robot_anchor,
                      const math::Quatf &motion_anchor,
                      std::vector<float> &value) const;
  bool AnchorOrientation(const std::array<float, 4> &root_quat_wxyz,
                         const std::array<float, kActionDim> &dof_pos,
                         math::Quatf &orientation) const;

  bool initialized_ = false;
  std::string model_path_;
  std::string anchor_body_name_ = "pelvis";
  std::unique_ptr<FkBody> fk_body_;

  std::array<float, kActionDim> default_dof_pos_{};
  std::array<float, kActionDim> kp_{};
  std::array<float, kActionDim> kd_{};
  std::array<float, kActionDim> action_scale_{};
  std::array<float, kActionDim> last_action_{};

  std::vector<Term> terms_;
  std::vector<int> trained_lookahead_offsets_;
  int motion_command_dim_ = 0;
  float observation_clip_ = 100.0f;
  std::uint64_t tick_ = 0;

  bool heading_initialized_ = false;
  math::Quatf robot_yaw_inv_ = math::Quatf::Identity();
  math::Quatf motion_yaw_inv_ = math::Quatf::Identity();

  std::function<double()> elapsed_seconds_fn_;
  double control_interval_s_ = 0.02;
  double engage_elapsed_s_ = 0.0;
  bool engage_anchored_ = false;
  float last_time_step_ = 0.0f;

  std::unique_ptr<holosoma::OnnxModel> session_;
  std::vector<float> obs_input_;
};

} // namespace holosoma::live
