// src/_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cadical.hpp"
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>

namespace py = pybind11;

// ---------- (optional) trampoline kept for when you add a Python propagator ----------
struct PyExternalProp : public CaDiCaL::ExternalPropagator {
    py::object pyprop;
    CaDiCaL::Solver* solver = nullptr;

    explicit PyExternalProp(py::object o) : pyprop(std::move(o)) {}
    void init(CaDiCaL::Solver* s) { solver = s; }

    void notify_new_decision_level() override {
        py::gil_scoped_acquire gil; pyprop.attr("notify_new_decision_level")();
    }
    void notify_backtrack(size_t new_level) override {
        py::gil_scoped_acquire gil; pyprop.attr("notify_backtrack")(py::int_(new_level));
    }
    void notify_assignment(const std::vector<int>& lits) override {
        py::gil_scoped_acquire gil; pyprop.attr("notify_assignment")(lits);
    }
    int  cb_propagate() override {
        py::gil_scoped_acquire gil; return pyprop.attr("cb_propagate")().cast<int>();
    }
    int  cb_add_reason_clause_lit(int propagated_lit) override {
        py::gil_scoped_acquire gil; return pyprop.attr("cb_add_reason_clause_lit")(propagated_lit).cast<int>();
    }
    int  cb_decide() override {
        py::gil_scoped_acquire gil; return pyprop.attr("cb_decide")().cast<int>();
    }
    bool cb_check_found_model(const std::vector<int>& model) override {
        py::gil_scoped_acquire gil; return pyprop.attr("cb_check_found_model")(model).cast<bool>();
    }
    bool cb_has_external_clause(bool &is_forgettable) override {
        py::gil_scoped_acquire gil;
        auto tup = pyprop.attr("cb_has_external_clause")().cast<std::pair<bool,bool>>();
        is_forgettable = tup.second; return tup.first;
    }
    int  cb_add_external_clause_lit() override {
        py::gil_scoped_acquire gil; return pyprop.attr("cb_add_external_clause_lit")().cast<int>();
    }
};

// ------------------------------- Solver wrapper ----------------------------
struct PySolver {
    std::unique_ptr<CaDiCaL::Solver> S{new CaDiCaL::Solver()};
    std::shared_ptr<PyExternalProp> ext; // keep trampoline alive
    py::object ext_keepalive;            // keep Python object alive
    int max_var = 0;

    PySolver() = default;

    void _update_max_from_lit(int lit) {
        int v = std::abs(lit);
        if (v > max_var) max_var = v;
    }

    void add_clause(const std::vector<int>& lits) {
        for (int lit : lits) {
            if (lit == 0) throw std::invalid_argument("literal 0 not allowed in add_clause()");
            _update_max_from_lit(lit);
            S->add(lit);
        }
        S->add(0); // end clause
    }

    void add_observed_var(int v) {
        if (v <= 0) throw std::invalid_argument("observed var must be positive");
        if (v > max_var) max_var = v;
        S->add_observed_var(v);
    }

    void connect_external_propagator(py::object pyprop) {
        ext = std::make_shared<PyExternalProp>(pyprop);
        S->connect_external_propagator(ext.get());
        ext->init(S.get());
        ext_keepalive = std::move(pyprop);
        py::gil_scoped_acquire gil;
        if (py::hasattr(ext_keepalive, "init"))
            ext_keepalive.attr("init")(py::cast(this, py::return_value_policy::reference));
    }

    int solve() {
        // DO NOT release GIL here; binding will do it via call_guard.
        return S->solve(); // 10 SAT, 20 UNSAT
    }

    std::vector<int> model() const {
        std::vector<int> m; m.reserve(static_cast<size_t>(max_var));
        for (int v = 1; v <= max_var; ++v) {
            int val = S->val(v); // >0 T, <0 F, 0 undef
            if (val > 0) m.push_back(+v);
            else if (val < 0) m.push_back(-v);
            else m.push_back(0);
        }
        return m;
    }
};

// --------------------------------- Module ----------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "cadical_py core bindings";

    py::class_<PySolver>(m, "Solver")
        .def(py::init<>())

        .def("add_clause", &PySolver::add_clause,
             R"doc(Add a CNF clause (list of DIMACS ints). 0 is implicit.)doc")
        .def("add_observed_var", &PySolver::add_observed_var,
             R"doc(Register a variable for external propagator notifications.)doc")
        .def("connect_external_propagator", &PySolver::connect_external_propagator,
             R"doc(Connect a Python ExternalPropagator instance.)doc")

        // Release GIL here (once), not inside the method:
        .def("solve", &PySolver::solve,
             R"doc(Solve: returns 10 (SAT) or 20 (UNSAT). GIL released during search.)doc",
             py::call_guard<py::gil_scoped_release>())

        .def("model", &PySolver::model,
             R"doc(Return model as list[int] of length max_var: +v / -v / 0.)doc")

        .def("__enter__", [](PySolver &self) -> PySolver& { return self; })
        .def("__exit__",  [](PySolver &, py::object, py::object, py::object) {});
}
