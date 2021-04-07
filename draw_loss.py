import matplotlib.pyplot as plt
import pandas as pd

log = pd.read_csv('log.csv')
log_heads = ['loss','loss0','loss1','loss2','loss3','loss4','loss5','loss6']

for i, head in enumerate(log_heads[1::]):
    plt.plot(log[head], label = head)

plt.legend()
plt.xlabel("epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)

plt.savefig("loss.png")