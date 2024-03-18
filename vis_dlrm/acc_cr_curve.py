import matplotlib.pyplot as plt
import re

def parse_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract data
    accuracy = []
    compression_ratios = []
    for line in lines:
        accuracy_match = re.search(r"accuracy\s(\d+\.\d+)\s%", line)
        compression_match = re.search(r"compression ratio is\s(\d+\.\d+)", line)

        if accuracy_match:
            acc = float(accuracy_match.group(1))
            accuracy.append(acc)

        elif compression_match:
            ratio = float(compression_match.group(1))
            compression_ratios.append(ratio)
    return accuracy, compression_ratios

# modification: minus min
def plot_accuracy(log_files):
    plt.figure(figsize=(10, 5))
    for i, file in enumerate(log_files):
        accuracy, _ = parse_log(file)
        plt.plot(accuracy, label=f'{file}')
    
    plt.title('Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(acc_png)
    plt.show()

def plot_compression_ratios(log_files):
    plt.figure(figsize=(10, 5))
    for i, file in enumerate(log_files):
        _, compression_ratios = parse_log(file)
        # Group data by every 26 appearances (as each group corresponds to a different table index)
        grouped_ratios = [compression_ratios[j:j+26] for j in range(0, len(compression_ratios), 26)]
        
        # Calculate the average compression ratio for each group
        avg_ratios = [sum(group) / len(group) for group in grouped_ratios]
        
        plt.plot(avg_ratios, label=f'{file}')
    
    plt.title('Average Compression Ratio Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Average Compression Ratio')
    plt.legend()
    plt.savefig(cr_png)
    plt.show()


# Example usage
'''
log_files = ['/N/u/haofeng/BigRed200/dlrm/decay_logs/step_2.log',
             '/N/u/haofeng/BigRed200/dlrm/decay_logs/step_3.log',
             '/N/u/haofeng/BigRed200/dlrm/decay_logs/constant_2.log',
             '/N/u/haofeng/BigRed200/dlrm/decay_logs/constant_3.log']  # Replace with your log file paths
             
'''
acc_png = "/N/u/haofeng/BigRed200/dlrm/decay_logs/decay_ls_acc.png"
cr_png = "/N/u/haofeng/BigRed200/dlrm/decay_logs/decay_ls_avg_cr.png"



log_files = ['/N/u/haofeng/BigRed200/dlrm/decay_logs/decay_short.log', 
             '/N/u/haofeng/BigRed200/dlrm/decay_logs/decay_long.log']  # Replace with your log file paths

plot_accuracy(log_files)
plot_compression_ratios(log_files)
