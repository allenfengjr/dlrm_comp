import matplotlib.pyplot as plt

# The sizes of each segment
alltoall_size_fwd = [31.3, 5.03, 3.64]
alltoall_size_bwd = [31.3, 31.3, 31.3]
allreduce_size = [22.9, 22.9, 22.9]
kernel_computation_size = [14.5, 14.5, 14.5]

# The y position for the single bar
bar_position = [0]

# Prepare the legend data
legend_data = [
    ('alltoall_fwd', alltoall_size_fwd),
    ('alltoall_bwd', alltoall_size_bwd),
    ('all-reduce', allreduce_size),
    ('kernel computation', kernel_computation_size)
]

# The cumulative sizes to stack the segments
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 3))  # Adjusted figure size
subplots_titles = ['Original Breakdown', 'Optimized Breakdown on Kaggle Dataset', "Optimized Breakdown on Terabyte Dataset"]  # 标题列表
rows = ["alltoall_fwd", "alltoall_bwd", "all-reduce", "kernel computation"]
for i, ax in enumerate(axes):
    cumulative_sizes = 0
    if i == 0:
        plt.xticks([0, 20, 40, 60 ,80, 100])
    elif i == 1:
        plt.xticks([0, 20, 40, 60 ,80])
    elif i == 2:
        plt.xticks([0, 20, 40, 60 ,80])
    ax.barh(bar_position, alltoall_size_fwd[i], color="white", hatch='\\\\', edgecolor='black')
    cumulative_sizes += alltoall_size_fwd[i]
    ax.barh(bar_position, alltoall_size_bwd[i], left=cumulative_sizes, color="white", hatch='...', edgecolor='black')
    cumulative_sizes += alltoall_size_bwd[i]
    ax.barh(bar_position, allreduce_size[i], left=cumulative_sizes, color="white", hatch='///', edgecolor='black')
    cumulative_sizes += allreduce_size[i]
    ax.barh(bar_position, kernel_computation_size[i], left=cumulative_sizes, color='darkgrey', edgecolor='black')
    ax.set_xlim([0, 100])
    
    # 设置子图标题
    ax.set_title(subplots_titles[i])

    # 隐藏子图周围的包围框和y轴的刻度标签
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    if i == 1:
        fig.legend(rows, loc='lower center', bbox_to_anchor=(0.5, -.15), ncol=2, fontsize=10, frameon=False)


# Manually add legend with data values outside the plots
textstr = '\n'.join([f'{name}: {data[0]}%, {data[1]}%, {data[2]}%' for name, data in legend_data])
fig.text(0.75, -0.1, textstr, fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

fig.tight_layout()
# 显示图形
plt.show()

fig.savefig("comp_breakdown.pdf", dpi=300, bbox_inches='tight')