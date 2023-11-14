import pickle
import matplotlib.pyplot as plt

with open("runtime_res.pkl", "rb") as f:
    res = pickle.load(f)

res_states = res[0]
fig, ax = plt.subplots()
for elem in res_states:
    list_times = [time[0] for time in res_states[elem] if len(time) > 0]
    
    ax.plot(list_times, label=elem)
ax.legend(loc="upper left")
plt.savefig("runtimes_plot.pdf")

