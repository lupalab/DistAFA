import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def plot_dict(filename, value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig=fig)
    
def plot_prob(prob, filename):
    B, K = prob.shape
    N = int(np.sqrt(B))
    fig, axs = plt.subplots(N, N, figsize=(8,6))
    n = 0
    for i in range(N):
        for j in range(N):
            p = prob[n]
            axs[i,j].bar(range(K), p)
            axs[i,j].set_ylim(0., 1.)
            axs[i,j].set_yticks([])
            axs[i,j].set_xticks([])
            # axs[i,j].set_ylabel('prob.')
            # axs[i,j].set_yticks([0.,0.5,1.])
            # axs[i,j].set_xlabel('class')
            # axs[i,j].set_xticks(range(0, K, 5))
            n += 1
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig=fig)
      