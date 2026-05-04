#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <immtrack/errors.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

namespace py = pybind11;

namespace {

template <class Filter>
void bind_filter(py::module_& m, const char* name) {
    py::class_<Filter>(m, name)
        .def(py::init<>())
        .def(py::init<double, double, double>(),
             py::arg("alpha") = 1e-3,
             py::arg("beta") = 2.0,
             py::arg("kappa") = 0.0)
        .def("init", &Filter::init, py::arg("state"), py::arg("cov"))
        .def("predict", &Filter::predict, py::arg("dt"))
        .def("update", &Filter::update, py::arg("measurement"))
        .def_property_readonly(
            "state",
            [](const Filter& f) { return typename Filter::StateVec(f.state()); })
        .def_property_readonly(
            "covariance",
            [](const Filter& f) {
                return typename Filter::StateMat(f.covariance());
            })
        .def_property_readonly_static(
            "N", [](py::object) { return Filter::N; })
        .def_property_readonly_static(
            "M", [](py::object) { return Filter::M; });
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "immtrack core bindings";

    py::register_exception<immtrack::CovarianceNotPsd>(
        m, "CovarianceNotPsd", PyExc_RuntimeError);
    py::register_exception<immtrack::InvalidArgument>(
        m, "InvalidArgument", PyExc_ValueError);
    py::register_exception<immtrack::NumericalError>(
        m, "NumericalError", PyExc_RuntimeError);

    using UkfPosVxyzYawCV =
        immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
    bind_filter<UkfPosVxyzYawCV>(m, "UkfPosVxyzYawCV");
}
