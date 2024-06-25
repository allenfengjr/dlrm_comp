# 原始文件的路径
original_file_path = 'EMB_8_iter_20.bin.Kaggle'

# 输出文件的路径
output_file_path = 'EMB_5_iter_20.bin.Kaggle.128'

# 要复制和拼接的次数
copy_times = 128

# 打开原始文件读取内容
with open(original_file_path, 'rb') as original_file:
    content = original_file.read()

# 打开输出文件并写入复制的内容
with open(output_file_path, 'wb') as output_file:
    for _ in range(copy_times):
        output_file.write(content)

print(f'{copy_times} copies of {original_file_path} have been combined into {output_file_path}.')
