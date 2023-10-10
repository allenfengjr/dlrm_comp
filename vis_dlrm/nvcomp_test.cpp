#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nvcomp/lz4.h>

int main(int argc, char** argv) {
    // Define input and output file paths
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    // Read data from the input binary file
    std::ifstream input_stream(input_file, std::ios::binary);
    if (!input_stream.is_open()) {
        std::cerr << "Error opening input file: " << input_file << std::endl;
        return 1;
    }

    // Determine the size of the input data
    input_stream.seekg(0, std::ios::end);
    size_t input_size = input_stream.tellg();
    input_stream.seekg(0, std::ios::beg);

    // Allocate memory for input data and read from the file
    std::vector<int> h_data(input_size / sizeof(int));
    input_stream.read(reinterpret_cast<char*>(h_data.data()), input_size);
    input_stream.close();

    // Allocate GPU memory and copy data to the GPU
    int* d_data;
    cudaMalloc(&d_data, input_size);
    cudaMemcpy(d_data, h_data.data(), input_size, cudaMemcpyHostToDevice);

    // Compression
    size_t temp_bytes;
    nvcompDecompressGetTempSize(d_data, num_elements, NVCOMP_TYPE_INT, &temp_bytes);

    void* d_temp;
    cudaMalloc(&d_temp, temp_bytes);

    size_t compressed_bytes;
    nvcompLZ4CompressCompressAsync(d_data, num_elements, NVCOMP_TYPE_INT, d_temp, temp_bytes, &compressed_bytes, nullptr);

    void* d_compressed_data;
    cudaMalloc(&d_compressed_data, compressed_bytes);

    nvcompLZ4CompressCompressAsync(d_data, num_elements, NVCOMP_TYPE_INT, d_temp, temp_bytes, d_compressed_data, &compressed_bytes, nullptr);

    // Decompression
    nvcompStatus_t status;
    nvcompLZ4Decompressor* decompressor;
    status = nvcompDecompressAsync(&decompressor, d_compressed_data, compressed_bytes);
    if (status != nvcompSuccess) {
        std::cerr << "Failed to create decompressor" << std::endl;
        return -1;
    }

    size_t decompressed_bytes;
    nvcompDecompressGetOutputSize(decompressor, &decompressed_bytes);
    
    void* d_decompressed_data;
    cudaMalloc(&d_decompressed_data, decompressed_bytes);

    nvcompBatchedLZ4DecompressAsync(decompressor, d_temp, temp_bytes, d_compressed_data, compressed_bytes, d_decompressed_data, decompressed_bytes, nullptr);

    // Clean up
    nvcompDecompressAsync(decompressor);
    cudaFree(d_data);
    cudaFree(d_temp);
    cudaFree(d_compressed_data);
    cudaFree(d_decompressed_data);
    delete[] h_data;

    return 0;
}
