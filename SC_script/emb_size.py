import matplotlib.pyplot as plt
import numpy as np

# 第一个日志数据
embedding_sizes_1 = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145,
    5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4,
    7046547, 18, 15, 286181, 105, 142572
]

# 第二个日志数据
embedding_sizes_2 = [
    9980333, 36084, 17217, 7378, 20134, 3, 7112, 1442, 61, 9758201,
    1333352, 313829, 10, 2208, 11156, 122, 4, 970, 14, 9994222,
    7267859, 9946608, 415421, 12420, 101, 36
]

# 创建两个子图
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# 对原始索引及大小进行处理
indices_1 = np.arange(len(embedding_sizes_1))
indices_2 = np.arange(len(embedding_sizes_2))

# 第一个子图：第一个日志数据
axs[0].bar(indices_1, embedding_sizes_1, color='white', edgecolor='black',hatch = '...')
axs[0].set_yscale('log')  # 使用对数尺度
axs[0].set_xlabel('EmbeddingTable Index')
axs[0].set_ylabel('Size (log scale)')
axs[0].set_title('Criteo Kaggle')

# 第二个子图：第二个日志数据
axs[1].bar(indices_2, embedding_sizes_2, color='grey', edgecolor = 'black', hatch = '///')
axs[1].set_yscale('log')
axs[1].set_xlabel('EmbeddingTable Index')
axs[1].set_ylabel('Size (log scale)')
axs[1].set_title('Criteo Terabytes')

plt.tight_layout()
plt.show()
plt.savefig('emb_size.pdf', dpi=300, bbox_inches='tight')