import os

input_directory = '/N/u/haofeng/BigRed200/SC_TB_emb/'

output_directory = '/N/slate/haofeng/SC_TB_emb/original_padding/'

copy_times = 128

# create output directory
os.makedirs(output_directory, exist_ok=True)

# iterate all files
for filename in os.listdir(input_directory):
    original_file_path = os.path.join(input_directory, filename)
    output_file_path = os.path.join(output_directory, f"{filename}.padding")

    # read original file
    with open(original_file_path, 'rb') as original_file:
        content = original_file.read()

    # write padding file
    with open(output_file_path, 'wb') as output_file:
        for _ in range(copy_times):
            output_file.write(content)

    print(f'{copy_times} copies of {filename} have been combined into {output_file_path}.')
