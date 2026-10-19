#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "holosoma_cpp/holosoma_inference_core.hpp"

namespace holosoma {
namespace live {
namespace {

using SonicRobotObs = RobotObservation;

#ifndef HOLOSOMA_LIVE_FIXTURE_DIR
#define HOLOSOMA_LIVE_FIXTURE_DIR "wbt_stub"
#endif
const char *kStubDir = HOLOSOMA_LIVE_FIXTURE_DIR;

std::string Model(const char *name) { return std::string(kStubDir) + "/" + name; }

std::array<float, 4> Yaw(float radians) { return {{std::cos(0.5f * radians), 0.0f, 0.0f, std::sin(0.5f * radians)}}; }

WbtReferenceFrame Frame(float marker, float yaw = 0.0f) {
  WbtReferenceFrame frame;
  for (int i = 0; i < kActionDim; ++i) {
    frame.dof_pos[i] = marker + 0.01f * i;
    frame.dof_vel[i] = -marker - 0.02f * i;
  }
  frame.root_quat_wxyz = Yaw(yaw);
  return frame;
}

WbtReferenceSequence Sequence(float current_marker, float future_base) {
  WbtReferenceSequence sequence;
  sequence.current = Frame(current_marker);
  for (int slot = 0; slot < 10; ++slot) sequence.lookahead.push_back(Frame(future_base + slot));
  return sequence;
}

SonicRobotObs Obs(float yaw = 0.0f) {
  SonicRobotObs obs;
  obs.root_quat_wxyz = Yaw(yaw);
  return obs;
}

void ExpectIdentityRot6d(const std::vector<float> &values, std::size_t offset) {
  const std::array<float, 6> expected{{1, 0, 0, 1, 0, 0}};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(values[offset + i], expected[i], 1e-4f) << "rot6d " << i;
}

void ExpectYawRot6d(const std::vector<float> &values, std::size_t offset, float yaw) {
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  const std::array<float, 6> expected{{c, -s, s, c, 0, 0}};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(values[offset + i], expected[i], 1e-4f) << "rot6d " << i;
}

TEST(WbtPolicyInit, DetectsAndLoadsMetadataCompleteSingleFileModels) {
  EXPECT_TRUE(WbtPolicy::IsWbtModelFile(Model("split_history.onnx")));
  EXPECT_FALSE(WbtPolicy::IsWbtModelFile(kStubDir));

  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  EXPECT_TRUE(policy.IsInitialized());
  EXPECT_EQ(policy.AnchorBodyName(), "pelvis");
  EXPECT_EQ(policy.LookaheadSlots(), 10);
  EXPECT_EQ(policy.TrainedLookaheadFrames(), 45);
  EXPECT_EQ(policy.LookaheadSampleOffsets(18), (std::vector<int>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}));
  EXPECT_FLOAT_EQ(policy.DefaultDofPos()[7], 0.07f);
  EXPECT_FLOAT_EQ(policy.Kp()[7], 47.0f);
  EXPECT_NEAR(policy.Kd()[7], 4.7f, 1e-6f);
}

TEST(WbtPolicyInit, SupportsMetadataNamedHumanoidsWithNonRootAnchor) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("named_humanoid.onnx")));
  EXPECT_FLOAT_EQ(policy.DefaultDofPos()[7], 0.07f);
  EXPECT_EQ(policy.AnchorBodyName(), "link_28");
}

TEST(WbtPolicyInit, RejectsAMetadataShapeMismatch) {
  WbtPolicy policy;
  EXPECT_FALSE(policy.Initialize(Model("bad_shape.onnx")));
  EXPECT_FALSE(policy.IsInitialized());
}

TEST(WbtPolicyInit, FailedReloadDisablesThePreviouslyLoadedModel) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("pelvis.onnx")));
  ASSERT_FALSE(policy.Initialize(Model("missing.onnx")));
  ASSERT_FALSE(policy.IsInitialized());
  std::array<float, kActionDim> action{};
  EXPECT_FALSE(policy.ComputeAction(Obs(), Sequence(1, 2), action.data()));
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  EXPECT_EQ(policy.LookaheadSlots(), 10);
}

TEST(WbtPolicyInit, GpuRequestCannotSilentlyUseACpuOnlyRuntime) {
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end())
    GTEST_SKIP() << "CPU-only SDK regression";
  WbtPolicy policy;
  EXPECT_FALSE(policy.Initialize(Model("split_history.onnx"), true));
  EXPECT_FALSE(policy.IsInitialized());
  EXPECT_TRUE(policy.Initialize(Model("split_history.onnx"), false));
}

#ifdef WBT_GPU_SMOKE_TEST
TEST(WbtPolicyGpu, MatchesCpuAcrossChangingReferences) {
  WbtPolicy cpu, gpu;
  ASSERT_TRUE(cpu.Initialize(Model("split_history.onnx"), false));
  ASSERT_TRUE(gpu.Initialize(Model("split_history.onnx"), true));
  std::array<float, kActionDim> cpu_action{}, gpu_action{};
  for (int tick = 0; tick < 20; ++tick) {
    const auto reference = Sequence(0.1f * tick, 0.2f * tick);
    ASSERT_TRUE(cpu.ComputeAction(Obs(), reference, cpu_action.data()));
    ASSERT_TRUE(gpu.ComputeAction(Obs(), reference, gpu_action.data()));
    for (int joint = 0; joint < kActionDim; ++joint) EXPECT_NEAR(cpu_action[joint], gpu_action[joint], 1e-4f);
  }
  EXPECT_GT(gpu_action[0], 0.0f);
}
#endif

TEST(WbtPolicyObservation, SplitHistoryKeepsProprioHistoryAndTrueFutureTargetsSeparate) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  std::array<float, kActionDim> action{};

  auto first = Sequence(1.0f, 10.0f);
  ASSERT_TRUE(policy.ComputeAction(Obs(), first, action.data()));
  const auto &obs1 = policy.LastObservation();
  ASSERT_EQ(obs1.size(), 894u);
  // motion_command has history 2: zero-padded oldest, current newest.
  EXPECT_FLOAT_EQ(obs1[0], 0.0f);
  EXPECT_FLOAT_EQ(obs1[58], first.current.dof_pos[0]);
  // actor_targets follows the complete 314-float proprio block and has
  // history 1.
  for (int slot = 0; slot < 10; ++slot) {
    const std::size_t offset = 314 + static_cast<std::size_t>(slot) * 58;
    EXPECT_FLOAT_EQ(obs1[offset], first.lookahead[slot].dof_pos[0]);
    EXPECT_FLOAT_EQ(obs1[offset + kActionDim], first.lookahead[slot].dof_vel[0]);
  }
  for (float value : action) EXPECT_FLOAT_EQ(value, 0.0f);

  auto second = Sequence(2.0f, 20.0f);
  ASSERT_TRUE(policy.ComputeAction(Obs(), second, action.data()));
  const auto &obs2 = policy.LastObservation();
  EXPECT_FLOAT_EQ(obs2[0], first.current.dof_pos[0]);
  EXPECT_FLOAT_EQ(obs2[58], second.current.dof_pos[0]);
  EXPECT_FLOAT_EQ(obs2[314], second.lookahead[0].dof_pos[0]);
  EXPECT_FLOAT_EQ(action[0], first.current.dof_pos[0]);

  policy.Reset();
  EXPECT_FALSE(policy.HeadingInitialized());
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(3.0f, 30.0f), action.data()));
  EXPECT_FLOAT_EQ(action[0], 0.0f) << "reset must clear observation history";
}

TEST(WbtPolicyHeading, ReanchorPreservesProprioAndActionHistory) {
  for (const char *name : {"split_history.onnx", "sonic_flat_0p2s.onnx"}) {
    SCOPED_TRACE(name);
    WbtPolicy resumed, continuous;
    ASSERT_TRUE(resumed.Initialize(Model(name)));
    ASSERT_TRUE(continuous.Initialize(Model(name)));
    std::array<float, kActionDim> action{}, control{};
    for (int tick = 0; tick < 12; ++tick) {
      auto reference = Sequence(0.1f * tick, 0.2f * tick);
      reference.lookahead.resize(resumed.LookaheadSlots());
      ASSERT_TRUE(resumed.ComputeAction(Obs(), reference, action.data()));
      ASSERT_TRUE(continuous.ComputeAction(Obs(), reference, control.data()));
    }
    const auto before = resumed.LastObservation();
    ASSERT_TRUE(std::any_of(before.begin(), before.end(), [](float value) { return value != 0.0f; }));
    ASSERT_GT(resumed.LastTimeStep(), 0.0f);
    resumed.ResetHeading();
    EXPECT_FALSE(resumed.HeadingInitialized());
    EXPECT_EQ(resumed.LastObservation(), before);
    auto next = Sequence(1.2f, 2.4f);
    next.lookahead.resize(resumed.LookaheadSlots());
    ASSERT_TRUE(resumed.ComputeAction(Obs(), next, action.data()));
    ASSERT_TRUE(continuous.ComputeAction(Obs(), next, control.data()));
    EXPECT_EQ(resumed.LastObservation(), continuous.LastObservation());
    EXPECT_EQ(action, control);
    EXPECT_FLOAT_EQ(resumed.LastTimeStep(), 0.0f);
    // A new activation still resets every history, unlike a heading reanchor.
    resumed.Reset();
    EXPECT_TRUE(std::all_of(resumed.LastObservation().begin(), resumed.LastObservation().end(),
                            [](float value) { return value == 0.0f; }));
  }
}

TEST(WbtPolicyObservation, FlatTargetsUseFunctionLeavesAndIndependentGroupHistories) {
  for (bool short_horizon : {false, true}) {
    SCOPED_TRACE(short_horizon);
    WbtPolicy policy;
    ASSERT_TRUE(policy.Initialize(Model(short_horizon ? "sonic_flat_0p2s.onnx" : "sonic_flat.onnx")));
    const int slots = short_horizon ? 5 : 10;
    const std::size_t base = short_horizon ? 837 : 930;
    EXPECT_EQ(policy.LookaheadSlots(), slots);
    EXPECT_EQ(policy.TrainedLookaheadFrames(), short_horizon ? 8 : 45);
    if (short_horizon) EXPECT_EQ(policy.LookaheadSampleOffsets(8), (std::vector<int>{0, 2, 4, 6, 8}));
    WbtReferenceSequence sequence;
    sequence.current = Frame(1, 0.3f);
    for (int i = 0; i < slots; ++i) sequence.lookahead.push_back(Frame(10 + i, 0.3f + 0.1f * i));
    std::array<float, kActionDim> action{};
    ASSERT_TRUE(policy.ComputeAction(Obs(0.8f), sequence, action.data()));
    const auto &values = policy.LastObservation();
    ASSERT_EQ(values.size(), short_horizon ? 2267u : 2680u);
    EXPECT_FLOAT_EQ(values[0], 0);  // old proprio is zero-padded
    EXPECT_FLOAT_EQ(values[(short_horizon ? 9 : 10) * 3 - 1],
                    -1);  // newest gravity
    for (int slot = 0; slot < slots; ++slot) {
      for (int joint = 0; joint < kActionDim; ++joint) {
        EXPECT_FLOAT_EQ(values[base + slot * 29 + joint], sequence.lookahead[slot].dof_pos[joint]);
        EXPECT_FLOAT_EQ(values[base + slots * 29 + slot * 29 + joint], sequence.lookahead[slot].dof_vel[joint]);
      }
      ExpectYawRot6d(values, base + slots * 58 + slot * 6, 0.1f * slot);
    }
    for (std::size_t i = base + slots * 64; i < values.size() - 3; ++i) EXPECT_FLOAT_EQ(values[i], 0);
    EXPECT_FLOAT_EQ(values[values.size() - 3], 1);
    EXPECT_FLOAT_EQ(values[values.size() - 2], 0);
    EXPECT_FLOAT_EQ(values.back(), 0);
    sequence.lookahead.pop_back();
    EXPECT_FALSE(policy.ComputeAction(Obs(), sequence, action.data()));
  }
}

TEST(WbtPolicyInit, FlatContractsRejectMissingMixedAndStackedFutureTerms) {
  WbtPolicy valid;
  ASSERT_TRUE(valid.Initialize(Model("sonic_flat_0p2s.onnx")));
  for (const char *name : {"flat_bad_missing.onnx", "flat_bad_mixed.onnx", "flat_bad_history.onnx"}) {
    SCOPED_TRACE(name);
    WbtPolicy invalid;
    EXPECT_FALSE(invalid.Initialize(Model(name)));
    EXPECT_FALSE(invalid.IsInitialized());
  }
}

TEST(WbtPolicyObservation, LegacyLookaheadUsesTheActorObsHistoryLength) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("legacy_history.onnx")));
  std::array<float, kActionDim> action{};
  auto first = Sequence(1.0f, 10.0f);
  ASSERT_TRUE(policy.ComputeAction(Obs(), first, action.data()));
  const auto &obs1 = policy.LastObservation();
  ASSERT_EQ(obs1.size(), 1474u);
  // Seven standard terms with history 2 occupy 314 floats. The lookahead term
  // then has its own zero-padded 580-float old frame and 580-float current
  // frame.
  EXPECT_FLOAT_EQ(obs1[314], 0.0f);
  EXPECT_FLOAT_EQ(obs1[894], first.lookahead[0].dof_pos[0]);
  EXPECT_FLOAT_EQ(obs1[894 + 9 * 58], first.lookahead[9].dof_pos[0]);

  auto second = Sequence(2.0f, 20.0f);
  ASSERT_TRUE(policy.ComputeAction(Obs(), second, action.data()));
  const auto &obs2 = policy.LastObservation();
  EXPECT_FLOAT_EQ(obs2[314], first.lookahead[0].dof_pos[0]);
  EXPECT_FLOAT_EQ(obs2[894], second.lookahead[0].dof_pos[0]);
}

TEST(WbtPolicyHeading, PelvisHeadingIsCapturedSeparatelyAndResetOnReengage) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("pelvis.onnx")));
  std::array<float, kActionDim> action{};
  WbtReferenceSequence reference;
  reference.current = Frame(0.0f, -0.4f);

  ASSERT_TRUE(policy.ComputeAction(Obs(1.0f), reference, action.data()));
  EXPECT_TRUE(policy.HeadingInitialized());
  ExpectIdentityRot6d(policy.LastObservation(), 58);

  reference.current.root_quat_wxyz = Yaw(0.2f);
  ASSERT_TRUE(policy.ComputeAction(Obs(1.0f), reference, action.data()));
  ExpectYawRot6d(policy.LastObservation(), 58, 0.6f);

  policy.ResetHeading();
  ASSERT_TRUE(policy.ComputeAction(Obs(1.0f), reference, action.data()));
  ExpectIdentityRot6d(policy.LastObservation(), 58);
}

// ── time_step comes from the clock, not the tick count ──────────────────────
//
// Python's TimestepUtil derives the motion phase from elapsed wall time over
// the control interval. A tick counter is equivalent ONLY while every tick
// lands on time, and diverges permanently the first time one is late or
// dropped — so these bind the value against a controllable fake clock rather
// than against the number of ComputeAction calls.

TEST(WbtPolicyTimeStep, PhaseFollowsElapsedTimeAcrossDroppedTicks) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  double now = 100.0;  // arbitrary steady_clock origin
  policy.SetControlInterval(0.02);
  policy.SetElapsedSecondsFn([&now]() { return now; });
  std::array<float, kActionDim> action{};

  // First tick anchors the phase at 0 regardless of the clock's origin.
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 0.0f);

  now += 0.02;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 1.0f);

  // 10 control intervals of wall time pass but only ONE inference runs — the
  // dropped-tick case. The phase must reflect elapsed time (10), not the
  // number of ticks taken (which a counter would report as 2).
  now += 0.20;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 11.0f) << "phase must track the clock, not the tick count";
}

TEST(WbtPolicyTimeStep, EngageAndUnpauseReanchorThePhase) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  double now = 5.0;
  policy.SetControlInterval(0.02);
  policy.SetElapsedSecondsFn([&now]() { return now; });
  std::array<float, kActionDim> action{};

  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  // Step off the start before jumping: while the phase is still 0 any advance
  // of more than one step is treated as a clock jump and re-anchors (see
  // AClockJumpBeforeTheFirstStepReanchors), so a bare +1.0s here would report 0
  // for that reason rather than the one under test.
  now += 0.02;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  now += 0.98;  // 50 frames in total
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 50.0f);

  // Reset is engage / unpause / return-to-manipulation. The motion restarts
  // from the top, so a stale anchor would resume mid-motion.
  policy.Reset();
  now += 7.5;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 0.0f) << "reset must re-anchor the phase, not resume mid-motion";
}

TEST(WbtPolicyTimeStep, AClockJumpBeforeTheFirstStepReanchors) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  double now = 0.0;
  policy.SetControlInterval(0.02);
  policy.SetElapsedSecondsFn([&now]() { return now; });
  std::array<float, kActionDim> action{};

  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  // Mirrors TimestepUtil's guard: a large jump while still at the start
  // re-anchors rather than starting the policy deep into the motion.
  now += 3.0;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 0.0f);

  // Having re-anchored, normal single-step advance resumes from there.
  now += 0.02;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 1.0f);
  now += 0.02;
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 2.0f);
}

TEST(WbtPolicyTimeStep, WithNoClockWiredThePhaseStillAdvances) {
  // The unit-test / opt-out path: no clock injected. Falling back to the tick
  // count keeps behaviour unchanged instead of freezing the phase at 0.
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  std::array<float, kActionDim> action{};
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 0.0f);
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 1.0f);
}

TEST(WbtPolicyTimeStep, ANonFiniteClockHoldsTheLastPhase) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  double now = 1.0;
  policy.SetControlInterval(0.02);
  policy.SetElapsedSecondsFn([&now]() { return now; });
  std::array<float, kActionDim> action{};
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  // One step at a time off the start, so the jump guard does not re-anchor.
  for (int i = 0; i < 3; ++i) {
    now += 0.02;
    ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  }
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 3.0f);

  // Holding is the conservative choice: feeding NaN into the graph would make
  // every downstream action non-finite and trip the finite check instead.
  now = std::nan("");
  ASSERT_TRUE(policy.ComputeAction(Obs(), Sequence(1.0f, 10.0f), action.data()));
  EXPECT_FLOAT_EQ(policy.LastTimeStep(), 3.0f);
}

TEST(WbtPolicyTimeStep, RejectsANonPositiveControlInterval) {
  WbtPolicy policy;
  EXPECT_THROW(policy.SetControlInterval(0.0), std::invalid_argument);
  EXPECT_THROW(policy.SetControlInterval(-0.02), std::invalid_argument);
  EXPECT_THROW(policy.SetControlInterval(std::nan("")), std::invalid_argument);
}

// ── the rescale is reported, and trained offsets are exposed ────────────────

TEST(WbtPolicyLookahead, TrainedOffsetsAreExposedForTheResolvedOffsetLog) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("split_history.onnx")));
  // sonic_aligned: the launch log compares these against the sampled offsets
  // so a compression or dilation is visible rather than inferred.
  EXPECT_EQ(policy.TrainedLookaheadOffsets(), (std::vector<int>{0, 5, 10, 15, 20, 25, 30, 35, 40, 45}));
  EXPECT_EQ(policy.LookaheadSampleOffsets(45), policy.TrainedLookaheadOffsets())
      << "sampling at the trained horizon must be trained-faithful";
}

TEST(WbtPolicyHeading, TorsoHeadingUsesFkForRobotAndReferenceAndResets) {
  WbtPolicy policy;
  ASSERT_TRUE(policy.Initialize(Model("torso.onnx")));
  EXPECT_EQ(policy.AnchorBodyName(), "link_1");
  std::array<float, kActionDim> action{};
  SonicRobotObs robot = Obs(0.3f);
  robot.dof_pos_mj[0] = 0.1f;
  robot.dof_pos_mj[1] = 0.1f;
  WbtReferenceSequence reference;
  reference.current = Frame(0.0f, -0.5f);
  reference.current.dof_pos.fill(0.0f);
  reference.current.dof_pos[0] = 0.4f;
  reference.current.dof_pos[1] = 0.2f;

  ASSERT_TRUE(policy.ComputeAction(robot, reference, action.data()));
  ExpectIdentityRot6d(policy.LastObservation(), 58);

  reference.current.dof_pos[0] = 1.0f;
  reference.current.dof_pos[2] = 0.7f;  // downstream of link_1; must not matter.
  ASSERT_TRUE(policy.ComputeAction(robot, reference, action.data()));
  ExpectYawRot6d(policy.LastObservation(), 58, 0.6f);

  policy.Reset();
  ASSERT_TRUE(policy.ComputeAction(robot, reference, action.data()));
  ExpectIdentityRot6d(policy.LastObservation(), 58);
}

}  // namespace
}  // namespace live
}  // namespace holosoma
