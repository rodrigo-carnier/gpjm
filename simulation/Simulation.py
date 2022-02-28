# -*- coding: utf-8 -*-

import numpy as np
import gpflow
import tensorflow as tf

set_default_float(np.float64)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline

from GPJMv4 import *
from GPJMv3_datagen import *
from GPJMv3_functions import *

N_t = 100
ts_N = np.linspace(0, N_t, 33).reshape(-1, 1)   # 2022-02 RMC upd10: problems with number type, will check later
ts = np.linspace(0, N_t, 3*33).reshape(-1, 1)   # 2022-02 RMC upd10: problems with number type, will check later

ss = np.array([(x,y,z) for x in range(3) for y in range(3) for z in range(3)], dtype = np.float64)

# A dataset is generated using a fixed random seed.
yn0, KIn0, yb, KIb, cs, Kcc, Kcc_conv, Kss = data_gen(ts, ss, [1, 0.75, 0.75], [0.5,0.75], [0.75, 3, 3], [0.015, 0.015], [1, 1])
yn_new, yn_idx, M = downsample_neuraldata(yn0, ts_N, ts)
yn = yn_new.T.ravel().reshape(-1, 1) #yn_new, yn: The actual neural data used in the simulation

# Latent states
plt.plot(cs)
plt.title("Latent states")
plt.show()

# Generated neural data (not downsampled)
plt.plot(yn0)
plt.title("Generated neural data (from the HRF-convolved kernel)")
plt.show()

# "Observed" neural data (downsampled)
plt.plot(yn_new)
plt.title("Observed neural data (downsampled)")
plt.show()

# Observed beavioral data
plt.plot(yb.T)
plt.title("Observed behavioral data")
plt.show()

# Initialize the model.
# Note that the noise parameter of the dynamics kernel and the variance parameter of the spatial kernel are fixed.
test2 = GPJMv4(yn, yb.T, ts_N, ts, 2, ss)
test2.likelihood_tX.variance = 1e-6
gpflow.set_trainable(test2.likelihood_tX, False)        # 2022-02 RMC upd11: trainable attributes are not assigned directly anymore, but need a method
gpflow.set_trainable(test2.kern_XN.kernel_s, False)     # 2022-02 RMC upd11: trainable attributes are not assigned directly anymore, but need a method

# Check the initialized model.
test2

# Fit the model to the simulated data.
opt = gpflow.optimizers.Scipy()                                         # 2022-02 RMC upd12: Scipy is now one option of subclass gpflow.optimizers
opt.minimize(test2.training_loss, variables=test2.trainable_variables)  # 2022-02 RMC upd13: call to "minimize" changed
print("2-var done")

# Get the log likelihood.
llk2 = test2.log_marginal_likelihood()                                  # 2022-02 RMC upd14: call "compute_log_likelihood" changed
print([llk2])

# For color-coding the two-dimensional latent dynamics.
# Not a core component of the model and/or analysis.
import matplotlib.colors as colors
import matplotlib.cm as cm
ts_long = np.linspace(0,100,1000)
ts_long.shape

# Extract the estimated dynamics
latent2 = test2.X.value

# Time-series plot
plt.plot(latent2, label="Estimated", linewidth=3)
plt.plot(cs, linestyle=":", c="k", linewidth=3, label="Ground truth")
plt.legend()
plt.show()

# Two-dimensional plots
latent2_0 = np.interp(ts_long, ts.ravel(), latent2[:,0])
latent2_1 = np.interp(ts_long, ts.ravel(), latent2[:,1])

rainbow = plt.get_cmap('coolwarm')
my_norm = colors.Normalize(0, 1)
color_map = cm.ScalarMappable(norm=my_norm, cmap='coolwarm')
col_intensity = np.sin(ts.ravel()/16)**2
col_intensity2 = np.sin(ts_long/16)**2
plt.plot(latent2_0, latent2_1, c="k", linestyle=":", linewidth=3)
for idx in range(ts_long.shape[0]):
    my_col = color_map.to_rgba(col_intensity2[idx], alpha = 0.5)
    plt.scatter(latent2_0[idx], latent2_1[idx], color=my_col, s=125)
plt.title("Two-dimensional latent dynamics")
plt.show()

plt.plot(np.sin(ts_long/8), np.sin(ts_long/4), c="k", linestyle=":", linewidth=3)
for idx in range(ts_long.shape[0]):
    my_col = color_map.to_rgba(col_intensity2[idx], alpha = 0.5)
    plt.scatter(np.sin(ts_long/8)[idx], np.sin(ts_long/4)[idx], color=my_col, s=125)
plt.title("Ground truth")
plt.show()

def recover_neural(m, ts_new):
    import tensorflow as tf
    from numpy.linalg import inv, cholesky
    ts = tf.Session().run(m.ts)
    ts_N = tf.Session().run(m.ts_N)
    ss = tf.Session().run(m.ss)
    Y_N = tf.Session().run(m.Y_N)
    Kstar = recover_Kxn(m, ts_new)
    KttI = recover_Kxn(m, ts) + np.eye(Y_N.shape[0], dtype = np.float64) * m.likelihood_XN.variance.read_value()
    L = cholesky(KttI)
    fmean = Kstar.T.dot(inv(L.T).dot((inv(L)).dot(Y_N)))
    v = inv(L).dot(Kstar)
    Vstar = Kstar - v.T.dot(v)
    sd = np.sqrt(np.diag(Vstar))
    return fmean, fmean.ravel().reshape(ss.shape[0], ts_N.shape[0]).T, Vstar, sd.ravel().reshape(ss.shape[0], ts_N.shape[0]).T

yhat2, yhat2_arr, yhat2_v, yhat2_sd = recover_neural(test2, ts)

fig, axs = plt.subplots(9, 3, figsize=(20, 25), sharex=True, sharey = True)
for r in range(9):
    for c in range(3):
        voxel_idx = c+r*3
        axs[r,c].plot(ts_N, yhat2_arr[:,voxel_idx], c="r", label="Mean prediction", linewidth=3)
        axs[r,c].plot(ts_N, yhat2_arr[:,voxel_idx] - 1.96 * yhat2_sd[:,voxel_idx], c="r", linestyle=":", label="95% predictive interval")
        axs[r,c].plot(ts_N, yhat2_arr[:,voxel_idx] + 1.96 * yhat2_sd[:,voxel_idx], c="r", linestyle=":")
        axs[r,c].plot(ts_N, yn_new[:,voxel_idx], c="k", label="Simulated data")
        axs[r,c].set_title("Voxel ("+str(int(ss[voxel_idx,0]))+","+str(int(ss[voxel_idx,1]))+","+str(int(ss[voxel_idx,2]))+")")
        if r == 0 and c == 2:
            axs[r,c].legend()
            
bhat2, _, bhat2_ci = recover_behavioral(test2, ts)

plt.figure(figsize=(20,5))
plt.plot(ts, yb.ravel(), c="k", linewidth = 5, label="Simulated data")
plt.plot(ts, bhat2, c="r", label = "Mean predicton", linewidth=2)
plt.plot(ts, bhat2_ci[:,0], c="r", linestyle = ":", label="95% predictive interval")
plt.plot(ts, bhat2_ci[:,1], c="r", linestyle = ":", linewidth=2)

plt.legend()
plt.title("Behavioral Data")
plt.show()

plt.imshow(Kss)
plt.title("Spatial Kernel: Ground Truth")
plt.show()

plt.imshow(test2.kern_XN.kernel_s.compute_K(ss, ss))
plt.title("Spatial Kernel: Estimated")
plt.show()

plt.imshow((M.T).dot(Kcc_conv.dot(M)))
np.save("figure-temporalkernel-truth.npy", (M.T).dot(Kcc_conv.dot(M)))
plt.title("Temporal kernel (after convolution, downsized): Ground Truth")
plt.show()
plt.imshow(test2.kern_XN.kernel_t.compute_K(latent2, latent2))
np.save("figure-temporalkernel-estimated.npy", test2.kern_XN.kernel_t.compute_K(latent2, latent2))
plt.title("Temporal kernel: Estimated")
plt.show()

Kss_est = test2.kern_XN.kernel_s.compute_K(ss, ss)
Kxx_est = test2.kern_XN.kernel_t.compute_K(latent2, latent2)
plt.imshow(np.kron(Kss_est, Kxx_est))
plt.title("Spatiotemporal Kernel: Estimated")

