#!/usr/bin/env julia



# Copyright 2023 L3Harris Technologies, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#



using HDF5
using JSON
using Plots
using Random
using SHA
using Dates
using UUIDs


using GraphicalModelLearning #GML




# First calculate the SHA1 sum of THIS JULIA SCRIPT
# this will be used to ensure we are calculating performance metrics with the latest version of this script
this_script_sha1sum = bytes2hex(SHA.sha1(read(open(@__FILE__, "r"))))
println("performance metrics calculation script SHA-1 hash:", this_script_sha1sum)




# go through each file in benchmark_performance_metrics, get each solution_uuid already processed (List A)
# go through each file in benchmark_performer_submissions get solution_uuid.
# If solution_uuid \in A **AND** it was processed with the current `this_script_sha1sum`, then skip it.  We assume it has already been processed.
# if it has been processed but with a different `this_script_sha1sum` then DELETE the old performance metrics file and re-process it.
# if solution_uuid is NOT in A, then process it.

if split(pwd(),"/")[end] != "scripts"
    # TODO: clean up this method to be more consistent when debugging in VS Code or just running the script.
    cd("scripts")
end
println("Current working directory: ", pwd())


benchmark_instances_relative_path = "../benchmark_instances/"
benchmark_performer_submissions_relative_path = "../benchmark_performer_submissions/"
benchmark_performance_metrics_relative_path = "../benchmark_performance_metrics/"

println("instances relative path: ", benchmark_instances_relative_path)
println("submissions/solutions relative path: ", benchmark_performer_submissions_relative_path)
println("performance metrics relative path: ", benchmark_performance_metrics_relative_path)


instance_files = readdir(benchmark_instances_relative_path)
submission_files = readdir(benchmark_performer_submissions_relative_path)
performance_metrics_files = readdir(benchmark_performance_metrics_relative_path)




# Build the list of solution_uuids that have already been processed.
processed_solution_uuids = []
for performance_metric_file in performance_metrics_files
    fname = benchmark_performance_metrics_relative_path * performance_metric_file
    if split(fname,".")[end] != "json"
        # skip over README.md and other supplementary files.
        continue
    end

    processed_solution_dict = JSON.parsefile(fname)
    solution_uuid = processed_solution_dict["solution_uuid"]
    if (this_script_sha1sum != processed_solution_dict["processed_with_perf_calculator_hash"])
        println("the file solution_uuid:$solution_uuid was already processed and the results are in $fname.  BUT the results in $fname were calculated with a different version of the performance calculator script.  The file $fname will be deleted.  The solution should be re-processed with the current version of the script.") #TODO: guarantee that we process it below!!

        rm(fname)
    else
        if (processed_solution_dict["solution_uuid"] in processed_solution_uuids)
            println("solution_uuid:$solution_uuid has already been processed. Skipping.")
        else
            push!(processed_solution_uuids, processed_solution_dict["solution_uuid"])
        end
    end
end
    












for submission_file in submission_files
    fname = benchmark_performer_submissions_relative_path * submission_file
    if split(fname,".")[end] != "h5"
        # skip over README.md and other supplementary files.
        continue
    end

    solution = HDF5.h5read(fname, "solution")
    solution_uuid = solution["solution_uuid"]
    if !(solution_uuid in processed_solution_uuids)
        println("processing solution_uuid:$solution_uuid ... ")

        benchmark_instance_uuid = solution["instance_uuid"]
        performance_metrics_uuid = string(uuid4())
        
        performance_metrics_dict = Dict()
        performance_metrics_dict["solution_uuid"] = solution_uuid
        performance_metrics_dict["instance_uuid"] = benchmark_instance_uuid      
        performance_metrics_dict["performance_metrics_uuid"] = performance_metrics_uuid
        performance_metrics_dict["processed_with_perf_calculator_hash"] = this_script_sha1sum
        performance_metrics_dict["timestamp"] = string(now(UTC))



        # locate and open corresponding benchmark instance 
        located_corresponding_benchmark_instance = false
        instance_fname = ""
        instance_data = Dict()
        for instance_file in instance_files
            instance_fname = benchmark_instances_relative_path * instance_file
            if split(instance_fname,".")[end] != "json"
                # skip over README.md and other supplementary files.
                continue
            end

            instance_data = JSON.parsefile(instance_fname)
            if benchmark_instance_uuid == instance_data["metadata"]["instance_uuid"]
                located_corresponding_benchmark_instance = true
                break
            end
        end
        @assert located_corresponding_benchmark_instance


        n = length(instance_data["graph_data"]["nodes"])

        time_limit_seconds = instance_data["benchmark_requirements"]["time_limit_seconds"]
        num_samples_k = instance_data["benchmark_requirements"]["num_samples_k"]
        
        

        # runtime metrics (average sample time)
        overall_time_seconds = solution["run_time"]["overall_time"]["seconds"]
        counts = solution["solution_reported"]["states_observed_counts"];
        num_samples_returned = sum(counts)
        performance_metrics_dict["returned_required_num_samples_k"] = (num_samples_returned >= num_samples_k)
        performance_metrics_dict["within_time_limit"] = (overall_time_seconds <= time_limit_seconds)        
        performance_metrics_dict["overall_time"] = overall_time_seconds
        performance_metrics_dict["average_seconds_per_sample"] = overall_time_seconds/num_samples_returned



        # parameter comparison norm metrics
        delta_parameter_1_norm = 999.9
        delta_parameter_0_norm = 999.9
        delta_parameter_2_norm = 999.9
        delta_parameter_inf_norm = 999.9

        performance_metrics_dict["delta_parameter_0_norm"] = delta_parameter_0_norm
        performance_metrics_dict["delta_parameter_1_norm"] = delta_parameter_1_norm
        performance_metrics_dict["delta_parameter_2_norm"] = delta_parameter_2_norm
        performance_metrics_dict["delta_parameter_inf_norm"] = delta_parameter_inf_norm


        # other probability-distribution-related metrics:
        performance_metrics_dict["total_variation"] = nothing
        performance_metrics_dict["kl_divergence"] = nothing




        # write out dictionary to JSON file
        output_filename = benchmark_performance_metrics_relative_path * "performance_metrics." * performance_metrics_uuid * ".json"
        open(output_filename, "w") do io
            JSON.print(io, performance_metrics_dict)
        end

    end





end

println("done")

