from kinematics.iscoord.coordination import RelativePhase
import numpy as np

# create signals
t = np.arange(0, 100, 0.01)
v0 = np.sin(t)  # first signal

phase_shift = np.pi/4
v1 = (np.sin(t-np.pi/4))  # second signal shifted
print(f"It should return {np.rad2deg(phase_shift)}\n")

c = RelativePhase()
relph = c.crp(v0, v1)
print(f"MARP = {c.marp}\nDPh = {c.dph}")

print(relph.shape)
# import matplotlib.pyplot as plt
# # plt.plot(relph)
# # plt.ylim([-1,1])
# plt.show()





