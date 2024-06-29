import statistics


EMB_num = 26
iter_num = 23 # 36 for kaggle, 23 for tb
B = 4

def speedup(CR, B, T_c, T_d):
    return 1/((1/CR)+ B*(1/T_c + 1/T_d))

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def read_a_file(filename):
    tmp = []
    with open(filename, 'r') as file:
        for line in file:
            number = float(line.strip())
            tmp.append(number)
    return split_list(tmp, iter_num)

lz_cr = read_a_file("./gpulz_cr_terabyte_all.txt")
huffman_cr = read_a_file("./huffman_cr_terabyte_all2.txt")

for i in range(len(lz_cr)):
    for j in range(len(lz_cr[0])):
        lz_cr[i][j] *= 4

hybrid_cr = [[0 for i in range(iter_num)] for i in range(EMB_num)]
comp_use  = [0 for i in range(EMB_num)]

# calculate max cr
for i in range(EMB_num):
    for j in range(iter_num):
        print("LZ speed up", speedup(lz_cr[i][j] , B, T_c=30, T_d=170), "LZ ratio, ", lz_cr[i][j])
        print("Huffman speed up", speedup(huffman_cr[i][j], B, T_c=60, T_d=40), "Huffman ratio, ", huffman_cr[i][j])
        if speedup(lz_cr[i][j], B, T_c=45, T_d=290) > speedup(huffman_cr[i][j], B, T_c=60, T_d=40):
            # 72 for 64MB, 32 for 16MB
            comp_use[i] = "lz"
            hybrid_cr[i][j] = lz_cr[i][j] * 4
        else:
            comp_use[i] = "huffman"
            hybrid_cr[i][j] = huffman_cr[i][j]


log_cr = [] # size = iter

for j in range(iter_num):
    avg_list = []
    for i in range(EMB_num):
        avg_list.append(hybrid_cr[i][j])
    avg_cr = statistics.harmonic_mean(avg_list)
    log_cr.append(avg_cr)

# plot with log_cr
print(log_cr)
print(comp_use)
