


The benchmark proctor shall calculate performance metrics for each solution file in the `../benchmark_performer_submissions` directory and place a corresponding performance metric file in this directory.  


The script to calculate performance metrics summary files is `../scripts/benchmark_proctor_evaluation.jl`
* Input to script: `../benchmark_performer_submissions/solution.<uuid-y>.h5`
* Output from script: `../benchmark_performance_metrics/performance_metrics.<uuid-z>.json`
