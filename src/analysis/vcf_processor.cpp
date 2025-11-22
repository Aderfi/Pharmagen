#include <iostream>
#include <string>
#include <vector>
#include <htslib/vcf.h>
#include <htslib/faidx.h>

using namespace std;

// --- CLASE PRINCIPAL ---
class VCFProcessor {
private:
    htsFile *vcf_file = nullptr;
    bcf_hdr_t *vcf_header = nullptr;
    faidx_t *fasta_index = nullptr;
    bcf1_t *record = nullptr;

public:
    VCFProcessor(const std::string& vcf_path, const std::string& fasta_path) {
        // 1. Cargar Índice Fasta
        fasta_index = fai_load(fasta_path.c_str());
        if (!fasta_index) {
            throw std::runtime_error("Error: No se pudo cargar el archivo FASTA o su índice (.fai).");
        }

        // 2. Abrir VCF
        vcf_file = bcf_open(vcf_path.c_str(), "r");
        if (!vcf_file) {
            throw std::runtime_error("Error: No se pudo abrir el archivo VCF.");
        }

        // 3. Leer Cabecera
        vcf_header = bcf_hdr_read(vcf_file);
        if (!vcf_header) {
            throw std::runtime_error("Error: No se pudo leer la cabecera del VCF.");
        }

        // 4. Inicializar estructura de registro
        record = bcf_init();
    }

    ~VCFProcessor() {
        // Limpieza de memoria manual (HTSlib es C puro)
        if (record) bcf_destroy(record);
        if (vcf_header) bcf_hdr_destroy(vcf_header);
        if (vcf_file) bcf_close(vcf_file);
        if (fasta_index) fai_destroy(fasta_index);
    }

    void run() {
        int sample_count = bcf_hdr_nsamples(vcf_header);
        if (sample_count == 0) {
            std::cerr << "Aviso: El VCF no contiene muestras." << std::endl;
            return;
        }

        // Solo procesamos la primera muestra (paciente actual)
        std::string sample_name = vcf_header->samples[0];
        std::cout << "Procesando Paciente: " << sample_name << "\n" << std::string(40, '-') << std::endl;

        // Buffers para genotipos
        int32_t *gt_arr = NULL, ngt_arr = 0;

        // --- BUCLE PRINCIPAL DE LECTURA ---
        while (bcf_read(vcf_file, vcf_header, record) == 0) {
            
            // Desempaquetar datos (lazy loading en C)
            bcf_unpack(record, BCF_UN_ALL);

            // Datos básicos
            std::string chrom = bcf_hdr_id2name(vcf_header, record->rid);
            int pos_0based = record->pos; // HTSlib usa 0-based
            int pos_1based = record->pos + 1; // Para mostrar al usuario
            std::string ref = record->d.allele[0];
            
            // Saltar si no hay alternativos (bloques homocigotos ref)
            if (record->n_allele < 2) continue;

            // --- VALIDACIÓN DE REFERENCIA (Seguridad) ---
            int len;
            // fetch_seq devuelve un char* que debemos liberar. Coordenadas 0-based inclusivas.
            char* ref_seq_ptr = faidx_fetch_seq(fasta_index, chrom.c_str(), pos_0based, pos_0based + ref.length() - 1, &len);
            
            if (!ref_seq_ptr) {
                std::cerr << "Error: No se pudo obtener secuencia para " << chrom << ":" << pos_1based << std::endl;
                continue;
            }

            std::string genome_ref(ref_seq_ptr);
            free(ref_seq_ptr); // Importante liberar memoria de C

            // Convertir a mayúsculas para comparar
            for (auto & c: genome_ref) c = toupper(c);

            if (genome_ref != ref) {
                std::cerr << "🚨 MISMATCH " << chrom << ":" << pos_1based 
                          << " VCF=" << ref << " FASTA=" << genome_ref << ". Saltando." << std::endl;
                continue;
            }

            // --- DECODIFICACIÓN DEL GENOTIPO ---
            int ngt = bcf_get_genotypes(vcf_header, record, &gt_arr, &ngt_arr);
            
            if (ngt <= 0) continue; // No hay info de genotipo

            // Asumimos diploide para la primera muestra (índices 0 y 1)
            // bcf_gt_allele decodifica la máscara de bits interna de VCF
            int allele1_idx = bcf_gt_allele(gt_arr[0]);
            int allele2_idx = bcf_gt_allele(gt_arr[1]);
            bool is_phased = bcf_gt_is_phased(gt_arr[1]);
            
            // Verificar si es nulo (./.)
            if (allele1_idx < 0 || allele2_idx < 0) continue;

            std::string a1 = record->d.allele[allele1_idx];
            std::string a2 = record->d.allele[allele2_idx];
            
            std::string tipo;
            if (allele1_idx == allele2_idx) {
                tipo = (allele1_idx == 0) ? "Homocigoto Ref" : "Homocigoto Alt";
            } else {
                tipo = (allele1_idx == 0 || allele2_idx == 0) ? "Heterocigoto" : "Heterocigoto Compuesto";
            }

            // Solo imprimir si no es WT (opcional)
            if (tipo != "Homocigoto Ref") {
                char sep = is_phased ? '|' : '/';
                std::cout << "📍 " << chrom << ":" << pos_1based << " (" << ref << "->" << record->d.allele[1] << ")\n";
                std::cout << "   └── " << tipo << " [" << a1 << sep << a2 << "]\n";
            }
        }

        // Liberar buffer de genotipos
        if (gt_arr) free(gt_arr);
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <archivo.vcf.gz> <referencia.fasta>" << std::endl;
        return 1;
    }

    try {
        // Instancia stack-allocated (RAII se encargará de limpiar al terminar)
        VCFProcessor processor(argv[1], argv[2]);
        processor.run();
    } catch (const std::exception& e) {
        std::cerr << "Excepción crítica: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}