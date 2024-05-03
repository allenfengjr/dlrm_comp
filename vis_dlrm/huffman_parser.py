import re

def parse_performance_data(log_data):
    results = []

    # Regular expressions to capture relevant data
    comp_hf_encode_pattern = re.compile(r"\[psz::info::res::comp_hf_encode\] shortest time \(ms\): .*highest throughput \(GiB/s\): ([\d.]+)")
    decomp_hf_decode_pattern = re.compile(r"\[psz::info::res::decomp_hf_decode\] shortest time \(ms\): .*highest throughput \(GiB/s\): ([\d.]+)")
    huffman_cr_pattern = re.compile(r"Huffman CR = sizeof\(E\) \* len / outlen\", where outlen is byte count:\s*([\d.]+)")

    # Finding all matches in the log data
    comp_hf_encodes = comp_hf_encode_pattern.findall(log_data)
    decomp_hf_decodes = decomp_hf_decode_pattern.findall(log_data)
    huffman_crs = huffman_cr_pattern.findall(log_data)

    print("Comp HF Encode Throughputs:", comp_hf_encodes)  # Diagnostic print
    print("Decomp HF Decode Throughputs:", decomp_hf_decodes)  # Diagnostic print
    print("Huffman CRs:", huffman_crs)  # Diagnostic print

    # Iterate over the found results (assuming all lists are of the same length)
    for encode, decode, cr in zip(comp_hf_encodes, decomp_hf_decodes, huffman_crs):
        results.append({
            "comp_hf_encode_throughput_GiB/s": float(encode),
            "decomp_hf_decode_throughput_GiB/s": float(decode),
            "huffman_CR": float(cr)
        })

    return results

# Example usage:
log_data = """
Executing: /N/u/haofeng/BigRed200/cusz-latest/build/example/bin_hf /N/slate/haofeng/SC_TB_emb/0.01_padding/EMB_0_iter_1.bin.quan.padding 2048 64 128 256
REVERT bklen to 256 for u1-type input.peeking data, 20 elements
    26     0    27     0    28     0    27     0    28     0    28
     0    29     0    26     0    29     0    28     0[psz::info::discard::comp_hf_encode] time (ms): 0.393120
[psz::info::discard::comp_hf_encode] time (ms): 0.370560
[psz::info::discard::comp_hf_encode] time (ms): 0.368256
[psz::info::discard::comp_hf_encode] time (ms): 0.361952
[psz::info::discard::comp_hf_encode] time (ms): 0.368448
[psz::info::discard::comp_hf_encode] time (ms): 0.367520
[psz::info::discard::comp_hf_encode] time (ms): 0.369248
[psz::info::discard::comp_hf_encode] time (ms): 0.367424
[psz::info::discard::comp_hf_encode] time (ms): 0.367744
[psz::info::discard::comp_hf_encode] time (ms): 0.368224
[psz::info::res::comp_hf_encode] shortest time (ms): 0.361952	highest throughput (GiB/s): 172.67
Huffman in  len:	16777216
Huffman out len:	5945440
"Huffman CR = sizeof(E) * len / outlen", where outlen is byte count:	2.82
[psz::info::discard::decomp_hf_decode] time (ms): 0.956128
[psz::info::discard::decomp_hf_decode] time (ms): 0.942080
[psz::info::discard::decomp_hf_decode] time (ms): 0.940032
[psz::info::discard::decomp_hf_decode] time (ms): 0.939008
[psz::info::discard::decomp_hf_decode] time (ms): 0.942080
[psz::info::discard::decomp_hf_decode] time (ms): 0.939008
[psz::info::discard::decomp_hf_decode] time (ms): 0.939008
[psz::info::discard::decomp_hf_decode] time (ms): 0.939008
[psz::info::discard::decomp_hf_decode] time (ms): 0.944128
[psz::info::discard::decomp_hf_decode] time (ms): 0.943104
[psz::info::res::decomp_hf_decode] shortest time (ms): 0.939008	highest throughput (GiB/s): 66.56
>>>>  IDENTICAL.
peeking xdata, 20 elements
    26     0    27     0    28     0    27     0    28     0    28
     0    29     0    26     0    29     0    28     0
Completed: EMB 0, Iter 1
Executing: /N/u/haofeng/BigRed200/cusz-latest/build/example/bin_hf /N/slate/haofeng/SC_TB_emb/0.01_padding/EMB_0_iter_2.bin.quan.padding 2048 64 128 256
REVERT bklen to 256 for u1-type input.peeking data, 20 elements
    31     0    29     0    30     0    28     0    30     0    29
     0    31     0    28     0    31     0    31     0[psz::info::discard::comp_hf_encode] time (ms): 0.349376
[psz::info::discard::comp_hf_encode] time (ms): 0.315584
[psz::info::discard::comp_hf_encode] time (ms): 0.310304
[psz::info::discard::comp_hf_encode] time (ms): 0.310944
[psz::info::discard::comp_hf_encode] time (ms): 0.312768
[psz::info::discard::comp_hf_encode] time (ms): 0.317984
[psz::info::discard::comp_hf_encode] time (ms): 0.309376
[psz::info::discard::comp_hf_encode] time (ms): 0.310336
[psz::info::discard::comp_hf_encode] time (ms): 0.314688
[psz::info::discard::comp_hf_encode] time (ms): 0.310560
[psz::info::res::comp_hf_encode] shortest time (ms): 0.309376	highest throughput (GiB/s): 202.02
Huffman in  len:	16777216
Huffman out len:	5891276
"Huffman CR = sizeof(E) * len / outlen", where outlen is byte count:	2.85
[psz::info::discard::decomp_hf_decode] time (ms): 0.948160
[psz::info::discard::decomp_hf_decode] time (ms): 0.934912
[psz::info::discard::decomp_hf_decode] time (ms): 0.935936
[psz::info::discard::decomp_hf_decode] time (ms): 0.935936
[psz::info::discard::decomp_hf_decode] time (ms): 0.933888
[psz::info::discard::decomp_hf_decode] time (ms): 0.935936
[psz::info::discard::decomp_hf_decode] time (ms): 0.942080
[psz::info::discard::decomp_hf_decode] time (ms): 0.934912
[psz::info::discard::decomp_hf_decode] time (ms): 0.933888
[psz::info::discard::decomp_hf_decode] time (ms): 0.935936
[psz::info::res::decomp_hf_decode] shortest time (ms): 0.933888	highest throughput (GiB/s): 66.92
>>>>  IDENTICAL.
peeking xdata, 20 elements
    31     0    29     0    30     0    28     0    30     0    29
     0    31     0    28     0    31     0    31     0
Completed: EMB 0, Iter 2"""

parsed_results = parse_performance_data(log_data)
for result in parsed_results:
    print(result)
