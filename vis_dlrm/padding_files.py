# # 原始文件的路径
# original_file_path = '/N/u/haofeng/BigRed200/SC_TB_emb/0.01/EMB_1_iter_1.bin.quan'

# # 输出文件的路径
# output_file_path = '/N/u/haofeng/BigRed200/SC_TB_emb/0.01/EMB_1_iter_1.bin.quan.128'

# # 要复制和拼接的次数
# copy_times = 128

# # 打开原始文件读取内容
# with open(original_file_path, 'rb') as original_file:
#     content = original_file.read()

# # 打开输出文件并写入复制的内容
# with open(output_file_path, 'wb') as output_file:
#     for _ in range(copy_times):
#         output_file.write(content)

# print(f'{copy_times} copies of {original_file_path} have been combined into {output_file_path}.')
import os

# 输入目录路径
input_directory = '/N/u/haofeng/BigRed200/SC_TB_emb/'

# 输出目录路径
output_directory = '/N/slate/haofeng/SC_TB_emb/original_padding/'

# 要复制和拼接的次数
copy_times = 128

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 遍历目录中的所有文件
for filename in os.listdir(input_directory):
    original_file_path = os.path.join(input_directory, filename)
    output_file_path = os.path.join(output_directory, f"{filename}.padding")

    # 读取原始文件内容
    with open(original_file_path, 'rb') as original_file:
        content = original_file.read()

    # 写入输出文件复制的内容
    with open(output_file_path, 'wb') as output_file:
        for _ in range(copy_times):
            output_file.write(content)

    print(f'{copy_times} copies of {filename} have been combined into {output_file_path}.')
