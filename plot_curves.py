import PAC.funcs as PAC
import matplotlib.pyplot as plt

N = 1000
beta = 0.99

test = PAC.calc_eps_risk_complexity(0.01, N*10, 1)

thresh = [0 for i in range(N)]
fixed = [0 for i in range(N)]
for k in range(N):
    thresh[k] = PAC.calc_eta_discard(beta, N, k)
    fixed[k] = PAC.calc_eta_fixed_discard(beta, N, k)

plt.plot(fixed)
plt.plot(thresh)

plt.show()
