import numpy as np
import matplotlib.pyplot as plt

# Function to draw a histogram on a given axes
def draw_histogram(data, table, iter, ax):
    ax.hist(data, bins=64, facecolor='white', alpha=0.7, edgecolor='black', hatch='....')
    ax.set_title(f"{table}", fontsize=10)
    ax.set_xlabel("Value", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.grid(which='major', color="lightgray", linestyle='--')

# Print the top n most frequent integers without using Counter
def print_top_frequent(data, n=20):
    value_counts = {}
    for val in data:
        if val in value_counts:
            value_counts[val] += 1
        else:
            value_counts[val] = 1
            
    most_common = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    print(f"Top {n} most frequent integers:")
    for val, freq in most_common:
        print(f"Value: {val}, Frequency: {freq}")

def main():
    iters = [2, 10, 20]
    table_name = ["Early phase", "Medium phase", "Late phase"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))  # 1 row, 3 columns of subplots

    for i, iter in enumerate(iters):  # Assuming tables 1, 2, 3
        filename = f"/home/haofeng/SC_emb_data/SC_TB_emb/EMB_19_iter_{iter}.bin"
        data = np.fromfile(filename, dtype=np.float32)
        
        draw_histogram(data, table_name[i], iter, axs[i])

    plt.tight_layout()
    plt.savefig(f"/home/haofeng/SC_script/hist_all_tables.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
