#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

namespace py = pybind11;

namespace {

template <class Filter>
void bind_filter(py::module_ &m, const char *name) {
  py::class_<Filter>(m, name)
      .def(py::init<>())
      .def("init", &Filter::init, py::arg("state"), py::arg("cov"))
      .def("predict", &Filter::predict, py::arg("dt"))
      .def("update", &Filter::update, py::arg("measurement"))
      .def_property_readonly("state",
                             [](const Filter &f) { return typename Filter::StateVec(f.state()); })
      .def_property_readonly(
          "covariance", [](const Filter &f) { return typename Filter::StateMat(f.covariance()); });
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "immtrack core bindings (v0.1 skeleton)";

  using CvBBox3DUKF = immtrack::UKF<immtrack::CV3D, immtrack::BBox3DObs>;
  using CtrvBBox3DUKF = immtrack::UKF<immtrack::CTRV3D, immtrack::BBox3DObs>;

  bind_filter<CvBBox3DUKF>(m, "CvBBox3DUKF");
  bind_filter<CtrvBBox3DUKF>(m, "CtrvBBox3DUKF");
}
