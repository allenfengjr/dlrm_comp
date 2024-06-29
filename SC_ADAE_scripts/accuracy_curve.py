import numpy as np
import matplotlib.pyplot as plt

file_path1 = 'accuracy_1.txt'
file_path2 = 'accuracy_2.txt'

with open(file_path1, 'r') as file1:
    # Read the file as a list of lines
    data_list1 = [float(line.strip()) for line in file1.readlines()]
    # Change the list to array
    global_eb1 = np.array(data_list1)

with open(file_path2, 'r') as file2:
    # Read the file as a list of lines
    data_list2 = [float(line.strip()) for line in file2.readlines()]
    # Change the list to array
    global_tb1 = np.array(data_list2)

# Input  data
global_eb = np.array([global_eb1, global_eb1, global_eb1])
global_tb = np.array([global_tb1, global_tb1, global_tb1])

rows = ['Global EB', 'Table-wise EB' ]
labels = ["Dataset1", "Dataset2", "Dataset3"]
colors = ['k', '#A60F2D']
linestyle_str = ['-.', '-.']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 2.5))

# Plot subfigures
for i, ax in enumerate(axes):
    ax.grid(True, color='w', linestyle='-', linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_facecolor('0.9')
    
    ax.plot(np.arange(len(global_eb[i])), global_eb[i], linestyle=linestyle_str[0], color=colors[0], linewidth = 1)
    ax.plot(np.arange(len(global_tb[i])), global_tb[i], linestyle=linestyle_str[1], color=colors[1], linewidth = 1)
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(labels[i], fontsize=10)

    # Add partial inset
    axins = inset_axes(ax, width="30%", height="30%", loc='center right')
    axins.plot(np.arange(len(global_eb[i])), global_eb[i], linestyle=linestyle_str[0], color=colors[0])
    axins.plot(np.arange(len(global_tb[i])), global_tb[i], linestyle=linestyle_str[1], color=colors[1])

    axins.set_xlim(250, 305)
    axins.set_ylim(78.5, 79.1)
    axins.grid(True, color='w', linestyle='-', linewidth=1)
    axins.set_facecolor('0.9')
    axins.tick_params(labelleft=True, labelbottom=True, labelsize='small')
    
    # Add the partial retangle to the main figure
    rect = Rectangle((247, 78.5), 55, 0.6, edgecolor='black', facecolor='none',
                     linestyle='--', linewidth=1, transform=ax.transData)
    ax.add_patch(rect)
    con = ConnectionPatch(xyA=(275, 78.6), coordsA=ax.transData,
                          xyB=(0.5, 1), coordsB=axins.transAxes,
                          arrowstyle="->", linestyle="--")
    ax.add_artist(con)


fig.legend(rows, fontsize='medium', loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

fig.tight_layout()
fig.savefig('/root/dlrm_comp/SC_ADAE_figures/accuracy_curve.pdf', dpi=300, bbox_inches='tight')