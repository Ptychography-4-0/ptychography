#include <pybind11/pybind11.h>
#include <string>

#include "algo.hpp"

namespace py = pybind11;

template<typename DATA_T,
	 typename SIZE_T>
void decare_algo_binding(py::module &m, const std::string &type_sufix){
  using Class = Algo<DATA_T, SIZE_T>;
  const std::string class_name = "Algo" + type_sufix;
  using BufferClass = typename Class::IOBuffer;
  const std::string buffer_name = class_name + "::IOBuffer";

  py::class_<Class>(m, class_name.c_str(), py::module_local())
    .def(py::init<size_t>())
    .def("init", &Class::init)
    .def("deinit", &Class::deinit)
    .def("get_size", &Class::get_size)
    .def("get_input", &Class::get_input)
    .def("get_output", &Class::get_output)
    .def("compute", &Class::compute);

  py::class_<BufferClass>(m, buffer_name.c_str(), py::module_local(), py::buffer_protocol())
    .def("data", &BufferClass::data, py::return_value_policy::reference)
    .def("get_size", &BufferClass::get_size)
    .def_buffer([](BufferClass &m) -> py::buffer_info {
		  return py::buffer_info(
					 m.data(),
					 sizeof(float),
					 py::format_descriptor<float>::format(),
					 1,
					 { m.get_size() },
					 { sizeof(float) }
					 );
		});

}

PYBIND11_MODULE(cppBinding, m) {
  decare_algo_binding<float, int>(m, "FI");
}
