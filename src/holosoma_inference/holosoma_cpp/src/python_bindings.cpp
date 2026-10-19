#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "holosoma_cpp/holosoma_inference_core.hpp"
#include "holosoma_cpp/inference_engine.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace py = pybind11;
using namespace holosoma;

namespace {
std::vector<float> observation(py::array_t<float, py::array::c_style | py::array::forcecast> a) {
  if (a.ndim() != 1 && !(a.ndim() == 2 && a.shape(0) == 1))
    throw py::value_error("Expected one observation with shape (N,) or (1,N)");
  return {a.data(), a.data() + a.size()};
}
py::array_t<double> array(const std::vector<double>& values) {
  py::array_t<double> out({py::ssize_t(1), static_cast<py::ssize_t>(values.size())});
  std::copy(values.begin(), values.end(), out.mutable_data());
  return out;
}
}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Transport-independent native inference; construction never opens a robot interface.";
  py::class_<RobotState>(m, "RobotState")
      .def(py::init<>())
      .def_readwrite("base_pos", &RobotState::base_pos)
      .def_readwrite("base_quat_wxyz", &RobotState::base_quat_wxyz)
      .def_readwrite("dof_pos", &RobotState::dof_pos)
      .def_readwrite("base_lin_vel", &RobotState::base_lin_vel)
      .def_readwrite("base_ang_vel", &RobotState::base_ang_vel)
      .def_readwrite("dof_vel", &RobotState::dof_vel)
      .def_readwrite("projected_gravity", &RobotState::projected_gravity)
      .def_readwrite("tick_ms", &RobotState::tick_ms);
  py::class_<JointCommand>(m, "JointCommand")
      .def_readonly("q", &JointCommand::q)
      .def_readonly("dq", &JointCommand::dq)
      .def_readonly("tau", &JointCommand::tau)
      .def_readonly("kp", &JointCommand::kp)
      .def_readonly("kd", &JointCommand::kd);
  py::class_<InferenceEngine>(m, "InferenceEngine")
      .def(py::init([](const std::string& config, bool autostart) {
             return std::make_unique<InferenceEngine>(config_from_yaml(config), autostart);
           }),
           py::arg("config"), py::arg("autostart") = true)
      .def("step",
           [](InferenceEngine& engine, py::array_t<float, py::array::c_style | py::array::forcecast> obs) {
             auto values = observation(obs);
             std::vector<double> result;
             {
               py::gil_scoped_release release;
               result = engine.step(values);
             }
             return array(result);
           })
      .def("step_state", &InferenceEngine::step_state, py::arg("state"), py::arg("advance_phase") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("set_state", &InferenceEngine::set_state)
      .def("infer_offsets",
           [](InferenceEngine& e, const RobotState& state) {
             std::vector<double> result;
             {
               py::gil_scoped_release release;
               result = e.infer_offsets(state);
             }
             return array(result);
           })
      .def("command",
           [](InferenceEngine& e, const std::string& name) {
             auto command = state_command_from_string(name);
             if (!command) throw py::value_error("Unknown state command: " + name);
             e.command(*command);
           })
      .def("velocity", [](InferenceEngine& e, double x, double y, double yaw) { e.velocity({x, y, yaw}); })
      .def("update_phase", &InferenceEngine::update_phase)
      .def("reset", &InferenceEngine::reset)
      .def_property_readonly("active", [](const InferenceEngine& e) { return e.policy().use_policy_action(); })
      .def_property_readonly("ready", [](const InferenceEngine& e) { return e.policy().get_ready_state(); })
      .def_property_readonly("active_index", [](const InferenceEngine& e) { return e.policy().active_index(); })
      .def_property_readonly("motion_timestep",
                             [](const InferenceEngine& e) -> std::optional<int> {
                               const auto* wbt = dynamic_cast<const WholeBodyTrackingPolicy*>(&e.policy());
                               return wbt ? std::optional<int>(wbt->motion_timestep()) : std::nullopt;
                             })
      .def_property_readonly("last_observation", [](const InferenceEngine& e) { return e.policy().last_observation(); })
      .def_property_readonly("joint_kp", [](const InferenceEngine& e) { return e.policy().joint_kp(); })
      .def_property_readonly("joint_kd", [](const InferenceEngine& e) { return e.policy().joint_kd(); });

  py::class_<live::RobotObservation>(m, "LiveRobotObservation")
      .def(py::init<>())
      .def_readwrite("root_quat_wxyz", &live::RobotObservation::root_quat_wxyz)
      .def_readwrite("dof_pos", &live::RobotObservation::dof_pos_mj)
      .def_readwrite("dof_vel", &live::RobotObservation::dof_vel_mj)
      .def_readwrite("base_ang_vel", &live::RobotObservation::base_ang_vel);
  py::class_<live::WbtReferenceFrame>(m, "ReferenceFrame")
      .def(py::init<>())
      .def_readwrite("dof_pos", &live::WbtReferenceFrame::dof_pos)
      .def_readwrite("dof_vel", &live::WbtReferenceFrame::dof_vel)
      .def_readwrite("root_quat_wxyz", &live::WbtReferenceFrame::root_quat_wxyz);
  py::class_<live::WbtReferenceSequence>(m, "ReferenceSequence")
      .def(py::init<>())
      .def_readwrite("current", &live::WbtReferenceSequence::current)
      .def_readwrite("lookahead", &live::WbtReferenceSequence::lookahead);
  py::class_<live::WbtPolicy>(m, "LiveWbtPolicy")
      .def(py::init([](const std::string& path, bool use_gpu) {
             auto policy = std::make_unique<live::WbtPolicy>();
             if (!policy->Initialize(path, use_gpu))
               throw py::value_error("Cannot initialize live WBT checkpoint: " + path);
             return policy;
           }),
           py::arg("model_path"), py::arg("use_gpu") = false)
      .def(
          "step",
          [](live::WbtPolicy& p, const live::RobotObservation& obs, const live::WbtReferenceSequence& ref,
             std::optional<double> elapsed_seconds) {
            if (elapsed_seconds)
              p.SetElapsedSecondsFn([value = *elapsed_seconds] { return value; });
            else
              p.SetElapsedSecondsFn({});
            std::array<float, live::kActionDim> offsets{};
            {
              py::gil_scoped_release release;
              if (!p.ComputeAction(obs, ref, offsets.data())) throw std::runtime_error("Live WBT inference failed");
            }
            std::vector<double> targets(live::kActionDim);
            for (int i = 0; i < live::kActionDim; ++i) targets[i] = offsets[i] + p.DefaultDofPos()[i];
            return array(targets);
          },
          py::arg("observation"), py::arg("reference"), py::arg("elapsed_seconds") = py::none())
      .def("set_control_interval", &live::WbtPolicy::SetControlInterval)
      .def("reset", &live::WbtPolicy::Reset)
      .def("reset_heading", &live::WbtPolicy::ResetHeading)
      .def("lookahead_offsets", &live::WbtPolicy::LookaheadSampleOffsets)
      .def_property_readonly("last_observation", &live::WbtPolicy::LastObservation)
      .def_property_readonly("kp", &live::WbtPolicy::Kp)
      .def_property_readonly("kd", &live::WbtPolicy::Kd);
}
