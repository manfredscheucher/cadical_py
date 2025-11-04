// cadipy.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cadical.hpp"

namespace py = pybind11;

struct PyExternalPropagator : public CaDiCaL::ExternalPropagator {
  py::object pyobj;

  explicit PyExternalPropagator(py::object obj) : pyobj(std::move(obj)) {}

  void notify_new_decision_level() override {
    py::gil_scoped_acquire gil;
    try { pyobj.attr("notify_new_decision_level")(); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); }
  }

  void notify_backtrack(size_t new_level) override {
    py::gil_scoped_acquire gil;
    try { pyobj.attr("notify_backtrack")(new_level); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); }
  }

  void notify_assignment(const std::vector<int>& lits) override {
    py::gil_scoped_acquire gil;
    try { pyobj.attr("notify_assignment")(lits); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); }
  }

  int cb_propagate() override {
    py::gil_scoped_acquire gil;
    try { return py::cast<int>(pyobj.attr("cb_propagate")()); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); return 0; }
  }

  int cb_add_reason_clause_lit(int propagated_lit) override {
    py::gil_scoped_acquire gil;
    try { return py::cast<int>(pyobj.attr("cb_add_reason_clause_lit")(propagated_lit)); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); return 0; }
  }

  int cb_decide() override {
    py::gil_scoped_acquire gil;
    try { return py::cast<int>(pyobj.attr("cb_decide")()); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); return 0; }
  }

  bool cb_check_found_model(const std::vector<int>& model) override {
    py::gil_scoped_acquire gil;
    try { return py::cast<bool>(pyobj.attr("cb_check_found_model")(model)); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); return false; }
  }

  bool cb_has_external_clause(bool &is_forgettable) override {
    py::gil_scoped_acquire gil;
    try { auto tup = py::cast<py::tuple>(pyobj.attr("cb_has_external_clause")()); 
          is_forgettable = py::cast<bool>(tup[1]); 
          return py::cast<bool>(tup[0]); 
    } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); is_forgettable = true; return false; }
  }

  int cb_add_external_clause_lit() override {
    py::gil_scoped_acquire gil;
    try { return py::cast<int>(pyobj.attr("cb_add_external_clause_lit")()); } catch (...) { PyErr_WriteUnraisable(pyobj.ptr()); return 0; }
  }
};

struct Solver {
  CaDiCaL::Solver* s;
  std::unique_ptr<PyExternalPropagator> keepalive;

  Solver(): s(new CaDiCaL::Solver()) {}
  ~Solver(){ delete s; }

  void connect_external_propagator(py::object pyprop){
    keepalive = std::make_unique<PyExternalPropagator>(pyprop);
    s->connect_external_propagator(keepalive.get());
  }
  void add_observed_var(int v){ s->add_observed_var(v); }
  int solve(){
    py::gil_scoped_release release;
    return s->solve();
  }
  // expose a bit more if you need it later
};

PYBIND11_MODULE(cadipy, m) {
  m.doc() = "Python bridge to CaDiCaL ExternalPropagator";
  py::class_<Solver>(m, "Solver")
    .def(py::init<>())
    .def("connect_external_propagator", &Solver::connect_external_propagator, py::arg("pyprop"))
    .def("add_observed_var", &Solver::add_observed_var)
    .def("solve", &Solver::solve);
}
