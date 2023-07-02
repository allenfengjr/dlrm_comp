import os
import matplotlib.pyplot as plt

filename = {
    "original":"/N/u/haofeng/BigRed200/dlrm/bigred_1616203.log", 
    
    "SZ5e-2":"/N/u/haofeng/BigRed200/dlrm/bigred_1617330.log",
    "ZFP3e-1":"/N/u/haofeng/BigRed200/dlrm/bigred_1617344.log",
    }
results = {}  # dictionary to store results for each log file

print(filename["original"])
for k,v in filename.items():
    print(k)
    f = open(filename[k],'r')
    lines = f.readlines()
    loss = []
    accuracy = []
    compression_ratio = []
    for l in lines:
        words = l.split(" ")
        if words[0] == "Finished":
            loss.append(float(words[-1].rstrip()))
        elif words[0] == "" and words[1]=="accuracy":
            accuracy.append(float(words[2]))
        elif words[0] == "Compression" and words[1] == "ratio,":
            compression_ratio.append(float(words[2].rstrip()))
    if k != "original":
        accuracy_delta = []
        for l in range(len(results["original"]["accuracy"])):
            accuracy_delta.append(results["original"]["accuracy"][l]-accuracy[l])
    else:
        accuracy_delta = []
    results[k] = {"loss": loss, "accuracy": accuracy, "compression_ratio": compression_ratio,"accuracy_delta": accuracy_delta}

    print(loss)
    print(accuracy)
    print(compression_ratio)

fig, axs = plt.subplots(2, figsize=(16, 16))

metrics = ["compression_ratio", "accuracy_delta"]
num_files = len(filename)

for i, metric in enumerate(metrics):
    axs[i].set_title(metric.capitalize())
    for j, (k, v) in enumerate(results.items()):
        axs[i].plot(v[metric], label=k)
    axs[i].legend()
    if i == len(metrics) - 1:
        for j in range(num_files):
            axs[i].plot([0, len(v[metric])], [0.02, 0.02], linestyle="--", color="gray")
            axs[i].plot([0, len(v[metric])], [0.0, 0.0], linestyle="--", color="gray")

plt.tight_layout()
plt.show()
plt.savefig("./batch_size_128_result_2.png")