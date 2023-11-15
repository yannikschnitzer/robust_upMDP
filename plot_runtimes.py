import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("runtime_res.pkl", "rb") as f:
    res = pickle.load(f)

res_states = res[0]

MDP_sizes = list(range(1,50,3))
MDP_sizes = [(num_states*2+2) for num_states in MDP_sizes]

fig, ax = plt.subplots()
for elem in res_states:
    list_times = [time for time in res_states[elem] if len(time) > 0]
    list_times = [time for time in list_times if sum(time) < 3600*5]
    #list_times = [[time for time in times if time < 3600] for times in list_times] 
    
    x_vals = MDP_sizes[:len(list_times)] 
    list_times = np.array(list_times).T

    min_times = np.min(list_times, axis=0)
    max_times = np.max(list_times, axis=0)
    mean_times = np.mean(list_times, axis=0)
    ax.fill_between(x_vals, min_times, max_times, alpha=0.7)
    ax.loglog(x_vals, mean_times, marker='x', linestyle="--", label=elem)
    
    #ax.boxplot(list_times, vert=True, labels=x_vals)
    
    #ax.loglog(x_vals, list_times, marker='x', linestyle="--", label=elem)
ax.legend(loc="upper right")
ax.set_xlabel(r'MDP size $|\mathcal{S}|\cdot|\mathcal{A}|$')
ax.set_ylabel("runtime (seconds)")
ax.set_title("Runtime vs MDP size")
plt.savefig("runtimes_plot_states.pdf")

num_samples = list(range(100,1000,50))

fig, ax = plt.subplots()
for elem in res[1]:
    #list_times = [time for time in res[1][elem] if len(time) > 0]
    #list_times = [[time for time in times if time < 3600] for times in list_times] 
    #x_vals = num_samples[:len(list_times)] 
    #ax.loglog(x_vals, list_times, marker='x', linestyle="--", label=elem)
    list_times = [time for time in res[1][elem] if len(time) > 0]
    list_times = [time for time in list_times if sum(time) < 3600*5]
    #list_times = [[time for time in times if time < 3600] for times in list_times] 
    
    x_vals = num_samples[:len(list_times)] 
    list_times = np.array(list_times).T

    min_times = np.min(list_times, axis=0)
    max_times = np.max(list_times, axis=0)
    mean_times = np.mean(list_times, axis=0)
    ax.fill_between(x_vals, min_times, max_times, alpha=0.7)
    ax.loglog(x_vals, mean_times, marker='x', linestyle="--", label=elem)
ax.legend(loc="upper right")
ax.set_xlabel('Number of Samples')
ax.set_ylabel("runtime (seconds)")
ax.set_title("Runtime vs Sample Size")
plt.savefig("runtimes_plot_samples.pdf")
