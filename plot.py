import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["savefig.bbox"] = "tight"

with open("data/res_14-14-42", "rb") as file:
    data = pickle.load(file)

Kd = data["Kd"]
Kg = data["Kg"]
Y = data["Y"]
i_d = data["i_d"]
i_g = data["i_g"]
v = data["v0"]

# fig 1
plt.figure()
plt.title("value function - dirty capital")
plt.plot(Kd, v[:,25,20], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[25]), Y[20]))
plt.xlabel("$K_d$")
plt.ylabel("v")
plt.legend()
# plt.savefig("./figures/v-Kd.pdf")
plt.show()

# fig 2
plt.figure()
plt.title("value function - green capital")
plt.plot(Kg, v[25,:,20], label="$K_d = {:d}, Y = {:.2f}$".format(int(Kd[25]), Y[20]))
plt.xlabel("$K_g$")
plt.ylabel("v")
plt.legend()
# plt.savefig("./figures/v-Kg.pdf")
plt.show()

# fig 3
plt.figure()
plt.title("value function - temperature")
plt.plot(Y, v[25,25,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[25]), int(Kg[25])))
plt.xlabel("$Y$")
plt.ylabel("v")
plt.legend()
# plt.savefig("./figures/v-Y.pdf")
plt.show()

# fig 1
plt.figure()
plt.title("dirty investment - dirty capital")
plt.plot(Kd, i_d[:,25,20], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[25]), Y[20]))
plt.xlabel("$K_d$")
plt.ylabel("$i_d$")
plt.legend()
# plt.savefig("./figures/id-Kd.pdf")
plt.show()

# fig 2
plt.figure()
plt.title("dirty investment - green capital")
plt.plot(Kg, i_d[25,:,20], label="$K_d = {:d}, Y = {:.2f}$".format(int(Kd[25]), Y[20]))
plt.xlabel("$K_g$")
plt.ylabel("$i_d$")
plt.legend()
# plt.savefig("./figures/id-Kg.pdf")
plt.show()

# fig 3
plt.figure()
plt.title("dirty investment - temperature")
plt.plot(Y, i_d[25,25,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[25]), int(Kg[25])))
plt.xlabel("$Y$")
plt.ylabel("$i_d$")
plt.legend()
# plt.savefig("./figures/id-Y.pdf")
plt.show()
# print(i_g)

# fig 1
plt.figure()
plt.title("green investment - dirty capital")
plt.plot(Kd, i_g[:,25,20], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[25]), Y[20]))
plt.xlabel("$K_d$")
plt.ylabel("$i_g$")
plt.legend()
# plt.savefig("./figures/ig-Kd.pdf")
plt.show()

# fig 2
plt.figure()
plt.title("green investment - green capital")
plt.plot(Kg, i_g[25,:,20], label="$K_d = {:d}, Y = {:.2f}$".format(int(Kd[25]), Y[20]))
plt.xlabel("$K_g$")
plt.ylabel("$i_g$")
plt.legend()
# plt.savefig("./figures/ig-Kg.pdf")
plt.show()

# fig 3
plt.figure()
plt.title("green investment - temperature")
plt.plot(Y, i_g[25,25,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[25]), int(Kg[25])))
plt.xlabel("$Y$")
plt.ylabel("$i_g$")
plt.legend()
# plt.savefig("./figures/ig-Y.pdf")
plt.show()
