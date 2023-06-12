from coordination import RelativePhase
import numpy as np

# create signals
t = np.arange(0.01, 100, 0.01)
v0 = np.sin(t)  # first signal

phase_shift = np.pi
v1 = (np.sin(t-phase_shift))  # second signal shifted
print(f"It should return {np.rad2deg(phase_shift)}\n")

c = RelativePhase()
relph = c.crp(v0, v1, plots=True)
print(f"MARP = {c.marp}\nDPh = {c.dph}")


# import matplotlib.pyplot as plt
# # plt.plot(relph)
# # plt.ylim([-1,1])
# plt.show()





