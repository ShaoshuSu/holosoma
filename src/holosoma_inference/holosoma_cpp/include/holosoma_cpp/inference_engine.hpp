#pragma once

#include "holosoma_cpp/policy.hpp"

namespace holosoma {

// In-process library boundary. Owns policy state, never opens a device or thread.
// The caller supplies observations and consumes commands at its own cadence.
class InferenceEngine {
 public:
  explicit InferenceEngine(InferenceConfig config, bool autostart = true);
  ~InferenceEngine();
  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;

  std::vector<double> step(const std::vector<float>& actor_observation);
  std::vector<double> infer_offsets(const RobotState& state);
  JointCommand step_state(const RobotState& state, bool advance_phase = true);
  void set_state(const RobotState& state);
  void command(StateCommand command);
  void velocity(const VelCmd& command);
  void update_phase();
  void reset();
  const BasePolicy& policy() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace holosoma
