import matplotlib.pyplot as plt
import numpy as np

comp_throughput = [100+i*10 for i in range(15)]
comp_ratio = [5+i for i in range(10)]
original_net_speed = [10, 20, 40]

res = [[0 for _ in range(10)] for _ in range(15)]
def speed_up(comp_r, comp_t, original_speed):
    return 1/(original_speed/comp_t + (1/comp_r))

for osp in original_net_speed:
    for i in range(15):
        for j in range(10):
            res[i][j] = speed_up(comp_r=comp_ratio[j], comp_t=comp_throughput[i], original_speed=osp)
    plt.clf()
    plt.xticks(np.arange(len(comp_ratio)), comp_ratio)
    plt.yticks(np.arange(len(comp_throughput)), comp_throughput)
    plt.imshow(res, cmap='hot', interpolation='nearest')
    plt.ylabel("compression throughput(GB/s)")
    plt.xlabel("compression ratio")
    plt.colorbar()
    plt.title(f"Communication Speed-up network bandwidth {osp} GB/s")
    plt.show()
    plt.savefig(f"speed-up{osp}.png")
print(res)