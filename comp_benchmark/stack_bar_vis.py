import matplotlib.pyplot as plt

# 输入数据，每一行包括压缩时间、解压时间、传输压缩数据时间和原数据传输时间
data = """
172.358817, 66.812634, 46.333320, 200.000000
370.346229, 125.177611, 108.268814, 200.000000
416.184759, 125.854679, 120.348747, 200.000000
149.382066, 39.242836, 37.503715, 200.000000
227.542986, 59.378869, 57.848911, 200.000000
37.378494, 6.699996, 2.852180, 200.000000
401.724200, 118.415211, 114.310209, 200.000000
259.468054, 69.123677, 67.787511, 200.000000
52.069923, 13.947920, 9.503971, 200.000000
191.499234, 74.313109, 53.444712, 200.000000
224.302755, 80.742560, 62.457158, 200.000000
322.644758, 120.688611, 94.667117, 200.000000
48.552099, 12.945004, 8.403498, 200.000000
104.916789, 28.636577, 24.462309, 200.000000
333.967426, 105.639312, 95.670846, 200.000000
55.510059, 14.659533, 11.334228, 200.000000
45.088340, 11.369186, 6.105737, 200.000000
72.814910, 19.423053, 17.027885, 200.000000
48.774026, 12.714733, 8.589034, 200.000000
187.241934, 70.800791, 50.962984, 200.000000
229.711480, 84.376935, 64.630880, 200.000000
199.641803, 76.409073, 55.206589, 200.000000
167.947885, 63.194991, 45.029924, 200.000000
406.364150, 118.684262, 116.310852, 200.000000
52.311277, 14.046408, 9.856238, 200.000000
58.478605, 16.093862, 11.522077, 200.000000
"""

# 解析数据
parsed_data = [list(map(float, line.split(','))) for line in data.strip().split('\n')]

# 数据分组
compression_times = [d[0] for d in parsed_data]
decompression_times = [d[1] for d in parsed_data]
transmission_compressed_times = [d[2] for d in parsed_data]
transmission_original_times = [d[3] for d in parsed_data]

# 创建图表
fig, ax = plt.subplots(figsize=(30, 8))  # 调整图表尺寸

# 设置数据
indices = range(len(parsed_data))  # x轴的索引
bar_width = 0.35  # 条形宽度

# 堆叠条形图
ax.bar(indices, compression_times, bar_width, label='Compression Time')
ax.bar(indices, decompression_times, bar_width, label='Decompression Time', bottom=compression_times)
ax.bar(indices, transmission_compressed_times, bar_width, label='Transmission Time Compressed', bottom=[i+j for i,j in zip(compression_times, decompression_times)])
ax.bar([i+bar_width for i in indices], transmission_original_times, bar_width, label='Transmission Time Original', color='red')

# 添加图例和标签
ax.set_xlabel('EMB Table Index')
ax.set_ylabel('Time (ms)')
ax.set_title('Time Comparisons for Each EMB Table')
ax.legend()

# 显示图表
# plt.show()
plt.savefig('lz4_breakdown.png')