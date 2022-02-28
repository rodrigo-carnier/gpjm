# GPJM v4: updating GPJM to conform to GPflow v2

GPflow changed significantly from v1 to v2. GPJM is built on top of GPflow and completely broke with the update.
This project updates the entire code of GPJM to make it compatible to GPflow v2.

# Gaussian Process Linking Functions for Mind, Brain, and Behavior

This repository provides codes and data used for "Gaussian Process Linking Functions for Mind, Brain, and Behavior" (Bahg, Evans, Galdo, & Turner, in press).

 * simulation: The code used for the simulated study on a spatiotemporal GPJM.
 * fMRI_data: Raw fMRI data and general linear model regressors.
 * fMRI_model: The code of the GPJM with three-dimensional latent dynamics. Includes pre-processed data files used for fitting the model.

---

LOG OF CHANGES:

--- 2022-02 RMC ---

GPJMv4.py - upd18: TODO - use different model since there is no abstract method defined for predict_f. This line is not correct...

GPJMv4.py - upd17: gpflow does not have attribute "settings" anymore.

Change generated error: "TypeError: Input 'y' of 'AddV2' Op has type float32 that does not match type float64 of argument 'x'."
    
Simulation.py - upd16: "gpflow.config.set_default_float" necessary

GPJMv4.py - upd15: ARD/ard argument is not passable directly, but directly by lengthscales. Trying different model (GPR) since there is no abstract method defined for predict_f.
    
Simulation.py - upd14: call "compute_log_likelihood" changed

Simulation.py - upd13: call to "minimize" changed

Simulation.py - upd12: Scipy is now one option of subclass gpflow.optimizers

Simulation.py - upd11: trainable attributes are not assigned directly anymore, but need a method

Simulation.py - upd10: problems with number type, will check later

Simulation.py - Including file "Simulation.py", with transcript of jupyter notebook of simulations, for easier documentation of updates.



GPJMv4.py - upd09: call to "PCA_reduce" changed according to upd08.

GPJMv4.py - upd08: method "pca_reduce" moved from "gpflow.models.gpvlm.PCA_reduce" to "gpflow.utilities.ops.pca_reduce".

GPJMv4.py - upd07: now when creating a kernel it is necessary to define abstract method "K_diag". For now I'm just copying method "K".

GPJMv4.py - upd06: "gpflow.models.Model" does not exist anymore. Basic template now is "gpflow.models.GPModel", but problems are arising.

GPJMv4.py - upd05: name of kernel "RBF" changed to "SquaredExponential".

GPJMv4.py - upd04: parameter "input_dim" removed from all kernel calls.

GPJMv4.py - upd03: Abstract function for calculating log_likelihood now is named like this. (Was this the purpose of this function "_build_likelihood"?). I may be mistaken, will check with author later.

GPJMv4.py - upd02: This is how name_scopes are defined nowaways.

GPJMv4.py - upd01: Parameters are now handled by GPflow. Remove lines "@gpflow.params_as_tensors"
