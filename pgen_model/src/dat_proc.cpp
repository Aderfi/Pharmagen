#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

namespace py = pybind11;

// Función para procesar columnas multi-label
// Entrada: Lista de strings crudos, mapa de clases, y separador
// Salida: Matriz binaria numpy (One-Hot Encoding)
py::array_t<float> fast_multi_label_binarize(const std::vector<std::string>& raw_data, 
                                             const std::unordered_map<std::string, int>& mapping,
                                             char separator) {
    
    size_t n_samples = raw_data.size();
    size_t n_classes = mapping.size();

    // Crear matriz de salida (n_samples x n_classes) inicializada en 0
    auto result = py::array_t<float>({n_samples, n_classes});
    auto ptr = result.mutable_unchecked<2>(); // Acceso directo a memoria sin chequeos Python

    // Inicializar a 0
    std::fill(ptr.mutable_data(0, 0), ptr.mutable_data(0, 0) + result.size(), 0.0f);

    for (size_t i = 0; i < n_samples; ++i) {
        std::stringstream ss(raw_data[i]);
        std::string segment;
        
        // Splitting usando el separador proporcionado
        while (std::getline(ss, segment, separator)) {
            // Trim de espacios
            size_t first = segment.find_first_not_of(' ');
            if (std::string::npos == first) continue;
            size_t last = segment.find_last_not_of(' ');
            std::string label = segment.substr(first, (last - first + 1));

            // Buscar en el mapa y marcar 1.0
            auto it = mapping.find(label);
            if (it != mapping.end()) {
                ptr(i, it->second) = 1.0f; 
            }
        }
    }

    return result;
}

PYBIND11_MODULE(fast_processor, m) {
    m.doc() = "Módulo C++ para acelerar preprocesamiento de farmaco-genética";
    m.def("multi_label_binarize", &fast_multi_label_binarize, "Fast MultiLabel Binarizer",
          py::arg("raw_data"), py::arg("mapping"), py::arg("separator") = ',');
}