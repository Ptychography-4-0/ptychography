#include <pybind11/pybind11.h>
#include <string>

#include "algo.hpp"

namespace py = pybind11;

template <typename DATA_T, typename SIZE_T>
void decare_algo_binding(py::module &m, const std::string &type_sufix) {
  using Class = Algo<DATA_T, SIZE_T>;
  const std::string class_name = "Algo" + type_sufix;

  py::class_<Class>(m, class_name.c_str(), py::module_local())
      .def(py::init<size_t>())
      .def("init", &Class::init)
      .def("deinit", &Class::deinit)
      .def("get_size", &Class::get_size)
      .def("get_input_view", &Class::get_input_view,
           py::return_value_policy::reference_internal)
      .def("get_output_view", &Class::get_output_view,
           py::return_value_policy::reference_internal)
      .def("compute", &Class::compute);
}

PYBIND11_MODULE(cppBinding, m) { decare_algo_binding<float, int>(m, "FI"); }
