
# Sampling

This work describes an example of generating Samples of state configurations from a 3D square-grid Ising Model.  This is a proposed benchmark in development for the DARPA Quantum Benchmarking program.  

*BEWARE:  This effort is in progress and many bugs and TODO's remain!*

License is TBD.  

The primary classes and routines used are in the `sampling.py` module.

The following Jupyter Notebooks demonstrate usage:
* `01_introduction.ipynb` demonstrates how to create the graph for our Ising Model.  Warning: right now the code is limited to *automatically* initializing square grid edges with periodic boundaries.  With some effort, the user can input custom topologies (e.g., triangular or other lattices.)
* `02_brute_force_probability_distribution_calculation.ipynb` explicitly calculates the partition function and the probability distribution function for our model.  Warning:  don't try this for graphs with number of nodes $n > 20$.
* `03_constructing_starting_sample_for_mcmc.ipynb` demonstrates a greedy randomized method to construct a starting sample for our Monte Carlo Markov Chain (MCMC).  The process is greedy because it incrementally sets the spin for each node, probabilistically favoring lower energy configurations.
* `04_mcmc.ipynb` shows running a MCMC with the Metropolis-Hastings proposal/acceptance method with random starting states.  We compare the resulting empirical probability distribution function (PDF) to the true PDF calculated in `02_brute_force_probability_distribution_calculation.ipynb`.  This notebook generates the `mcmc_samples.csv` file.
* `04a_low_temp_mcmc_random_starts.ipynb` is similar to the previous notebook, but uses a low temperature $T=0.5$ value.
* `04b_low_temp_mcmc_greedy_random_starts.ipynb` is similar to the previous notebook, but uses a low temperature $T=0.5$ value and uses the greedy randomized state construction method to generate starting states for the MCMC.  Converge is visually inspected and compared to the random start method.
* `05_verification_of_samples.ipynb` is not implemented yet, but will use the procedure from https://github.com/lanl-ansi/GraphicalModelLearning.jl to reverse engineer the parameters of the underlying Ising model from samples generated from `04_mcmc.ipynb`.




## Discussion and questions:

### Benchmark Workflow

A variety of models will be version controlled and precisely described including:
* number of nodes (some small, medium, large instances)
* edge (interaction) topology (may be square grid lattice, frustrated, ferromagnetic or antiferromagnetic)
* external field strength
* node interaction strength
* system temperature (fixed at specific values of $T$)
* etc.

The benchmark performer will select one model at a time.

The benchmark performer will generate $k$ samples from the model according to the algorithm/hardware of their choice.

The benchmark proctor will perform the verification process (https://github.com/lanl-ansi/GraphicalModelLearning.jl) to determine if the $k$ samples faithfully represent the underlying model.


### Utility Thresholds

Utility Thresholds establish the performance metrics of classical computing methods against the benchmark.  A benchmark performer that can beat the utility threshold(s) gets a gold star.

For sampling, we will run a Monte Carlo Markov Chain (MCMC) with Metropolis Hastings (M-H) proposal/acceptance rules until the chain has mixed and then take the last state as one sample.  While MCMC/M-H is rather old, it serves as a reasonable, general purpose procedure for sampling.  There are many newer methods that perform better by taking advantage of the topology of the system or other features, but we prefer to set the utility thresholds using MCMC/M-H due to its ability to perform reasonably well for a wide variety of system topologies/features.  If I were benchmarking linear programming optimization algorithms, I'd make a very similar argument to use the Simplex algorithm to set the utility threshold: while the Simplex algorithm is old and there are lots of shiny new customized heuristics and methods out there, the Simplex algorithm still gets it done!  *.... Argue with me! Opine why we should use something other than MCMC/M-H!*

### Issues with Python

For actually establishing the Utility Thresholds, we will implement MCMC/M-C in C++ or possibly Julia for a faster per sample time.  Python serves us now for discussion and examples.

### Issues with MCMC/M-H

While there seem to be a variety of diagnostics that tell you if your MCMC has *not* converged to the stationary distribution, there seems to be no diagnostics to confirm that your MCMC *has* converged.  Thus, we are left with the unsettling feeling of running our MCMC for what we surmise to be enough steps and hoping for the best.  We can perform visual inspection of the convergence, we can execute a suite of diagnostics to confirm that we *haven't* converged, but nothing explicitly confirms convergence.  

### Issues with verification of Samples

We intend to use the method in https://github.com/lanl-ansi/GraphicalModelLearning.jl to reverse engineer the properties of the original Ising Model (edge interaction strength, external field, etc.).  At low system $T$ temperatures, the interaction strength between nodes in the system is very strong and the number of samples required to estimate the parameters grows exponentially with the maximum interaction strength (See https://doi.org/10.1088/1742-5468/ac3aea).  

We propose to reconcile this as follows.  The benchmark performer will be required to produce a sufficient number of samples at higher temperatures so that verification can be performed.  The benchmark performer will also be required to produce samples from the system at lower temperatures using the same algorithm/hardware, but these will not be verified.

### Greedy Randomized Construction of Samples (starting samples for MCMC)

This was a side quest.  I was curious to see the benefit of *constructing* a state using a greedy, randomized procedure.  The construction procedure is "greedy" as it sets the spin state of each node one at a time while probabilistically favoring spins that produce a lower system energy.  The empirical distribution of samples constructed using this method starts to resemble the true PDF, but it's different enough to not immediately accept the constructed samples as is.  I didn't have much hope for this working since, if it did, it would have large implications for the polynomial hierarchy.  But I was curious how similar the distribution of states produced by construction was to the real PDF anyway.  

Theoretically any spin configuration is a feasible state in our MCMC and may eventually be visited.  So we should be able to start our MCMC in any random state--even high-energy states--and it will converge to its stationary distribution.  But then I was also curious to see if staring the MCMC/M-H procedure with a greedy, randomized constructed state would show an lower mixing time.  At higher system temperatures it didn't seem to help much.  At low system temperatures, it appeared to help with convergence per visual inspection of the plot of system energy per step of the MCMC.  As predicted, either way, the MCMC/M-H still gets "stuck" in certain deadlocked spin configurations at low temperatures.  The impatient user will never see the MCMC converge.  



