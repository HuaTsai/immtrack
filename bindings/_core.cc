#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <immtrack/bbox.hpp>
#include <immtrack/cost_policies.hpp>
#include <immtrack/errors.hpp>
#include <immtrack/metrics/amota.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/tracked_object.hpp>
#include <immtrack/tracker.hpp>
#include <immtrack/ukf.hpp>

#include <string>
#include <utility>
#include <vector>

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

    py::class_<immtrack::BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init([](double x, double y, double z, double l, double w,
                         double h, double rot, std::string class_name,
                         double score, int track_id) {
                 immtrack::BoundingBox b;
                 b.x = x;
                 b.y = y;
                 b.z = z;
                 b.l = l;
                 b.w = w;
                 b.h = h;
                 b.rot = rot;
                 b.class_name = std::move(class_name);
                 b.score = score;
                 b.track_id = track_id;
                 return b;
             }),
             py::arg("x") = 0.0, py::arg("y") = 0.0,
             py::arg("z") = 0.0, py::arg("l") = 0.0,
             py::arg("w") = 0.0, py::arg("h") = 0.0,
             py::arg("rot") = 0.0,
             py::arg("class_name") = std::string(),
             py::arg("score") = 0.0, py::arg("track_id") = -1)
        .def_readwrite("x", &immtrack::BoundingBox::x)
        .def_readwrite("y", &immtrack::BoundingBox::y)
        .def_readwrite("z", &immtrack::BoundingBox::z)
        .def_readwrite("l", &immtrack::BoundingBox::l)
        .def_readwrite("w", &immtrack::BoundingBox::w)
        .def_readwrite("h", &immtrack::BoundingBox::h)
        .def_readwrite("rot", &immtrack::BoundingBox::rot)
        .def_readwrite("class_name", &immtrack::BoundingBox::class_name)
        .def_readwrite("score", &immtrack::BoundingBox::score)
        .def_readwrite("track_id", &immtrack::BoundingBox::track_id);

    py::class_<immtrack::TrackedObject>(m, "TrackedObject")
        .def_readonly("id", &immtrack::TrackedObject::id)
        .def_readonly("class_name", &immtrack::TrackedObject::class_name)
        .def_readonly("x", &immtrack::TrackedObject::x)
        .def_readonly("y", &immtrack::TrackedObject::y)
        .def_readonly("z", &immtrack::TrackedObject::z)
        .def_readonly("vx", &immtrack::TrackedObject::vx)
        .def_readonly("vy", &immtrack::TrackedObject::vy)
        .def_readonly("vz", &immtrack::TrackedObject::vz)
        .def_readonly("rot", &immtrack::TrackedObject::rot)
        .def_readonly("l", &immtrack::TrackedObject::l)
        .def_readonly("w", &immtrack::TrackedObject::w)
        .def_readonly("h", &immtrack::TrackedObject::h)
        .def_readonly("score", &immtrack::TrackedObject::score)
        .def_readonly("age", &immtrack::TrackedObject::age)
        .def_readonly("hit_count", &immtrack::TrackedObject::hit_count)
        .def_readonly("miss_count", &immtrack::TrackedObject::miss_count);

    using Tracker =
        immtrack::BBoxTracker<UkfPosVxyzYawCV, immtrack::MahalanobisCost>;

    py::class_<Tracker>(m, "BBoxTracker")
        .def(py::init([](int n_init, int max_age, int min_hits,
                         double size_ema_alpha) {
                 Tracker::Config c;
                 c.n_init = n_init;
                 c.max_age = max_age;
                 c.min_hits = min_hits;
                 c.size_ema_alpha = size_ema_alpha;
                 return Tracker(c);
             }),
             py::arg("n_init") = 3, py::arg("max_age") = 5,
             py::arg("min_hits") = 3, py::arg("size_ema_alpha") = 0.7)
        .def("update",
             py::overload_cast<const std::vector<immtrack::BoundingBox>&>(
                 &Tracker::update),
             py::arg("detections"))
        .def("update",
             py::overload_cast<const std::vector<immtrack::BoundingBox>&,
                               double>(&Tracker::update),
             py::arg("detections"), py::arg("dt"))
        .def("reset", &Tracker::reset)
        .def("track_count", &Tracker::track_count);

    auto metrics = m.def_submodule("metrics", "AMOTA evaluation utilities");

    py::enum_<immtrack::metrics::MatchMetric>(metrics, "MatchMetric")
        .value("CenterDistance",
               immtrack::metrics::MatchMetric::CenterDistance)
        .value("Iou3d", immtrack::metrics::MatchMetric::Iou3d);

    py::class_<immtrack::metrics::AmotaConfig>(metrics, "AmotaConfig")
        .def(py::init<>())
        .def_readwrite("recall_values",
                       &immtrack::metrics::AmotaConfig::recall_values)
        .def_readwrite("metric", &immtrack::metrics::AmotaConfig::metric)
        .def_readwrite("match_threshold",
                       &immtrack::metrics::AmotaConfig::match_threshold);

    py::class_<immtrack::metrics::AmotaResult>(metrics, "AmotaResult")
        .def_readonly("overall", &immtrack::metrics::AmotaResult::overall)
        .def_readonly("per_class",
                      &immtrack::metrics::AmotaResult::per_class);

    metrics.def("amota", &immtrack::metrics::amota, py::arg("gt"),
                py::arg("pred"),
                py::arg("cfg") = immtrack::metrics::AmotaConfig{});
}
