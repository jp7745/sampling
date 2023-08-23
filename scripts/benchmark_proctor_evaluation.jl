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




# utility functions to switch between spin_config vector and state_index
function convert_state_index_to_spin_config(state_index, n)
    spin_config = zeros(n);
    uint_1 = UInt128(1); #TODO: ensure proper bit width wrt n
    for i in 0:n-1
        bit = (state_index & (uint_1<<i) != 0);
        spin_config[i+1] = 2.0*bit - 1.0;
    end
    return spin_config;
end;

# utility functions to switch between spin_config vector and state_index
function convert_spin_config_to_state_index(spin_config)
    state_index = UInt128(0) #TODO: ensure proper bit width wrt n
    uint_1 = UInt128(1);
    n = length(spin_config);
    for i in 1:n
        if spin_config[i] > 0
            state_index = state_index | (uint_1<<(i-1));
        end
    end

    # state_index will be zero for all spin down (-1)
    # state_index will be Int(2^n-1) for all spin up (+1)
    return state_index;
end;



















# First calculate the SHA1 sum of THIS JULIA SCRIPT
# this will be used to ensure we are calculating performance metrics with the latest version of this script
this_script_sha1sum = bytes2hex(SHA.sha1(read(open(@__FILE__, "r"))));
println("performance metrics calculation script SHA-1 hash:", this_script_sha1sum);




# go through each file in benchmark_performance_metrics, get each solution_uuid already processed (List A)
# go through each file in benchmark_performer_submissions get solution_uuid.
# If solution_uuid \in A **AND** it was processed with the current `this_script_sha1sum`, then skip it.  We assume it has already been processed.
# if it has been processed but with a different `this_script_sha1sum` then DELETE the old performance metrics file and re-process it.
# if solution_uuid is NOT in A, then process it.

if split(pwd(),"/")[end] != "scripts"
    # TODO: clean up this method to be more consistent when debugging in VS Code or just running the script.
    cd("scripts");
end
println("Current working directory: ", pwd());


benchmark_instances_relative_path = "../benchmark_instances/";
benchmark_performer_submissions_relative_path = "../benchmark_performer_submissions/";
benchmark_performance_metrics_relative_path = "../benchmark_proctor_performance_metrics/";

println("instances relative path: ", benchmark_instances_relative_path);
println("submissions/solutions relative path: ", benchmark_performer_submissions_relative_path);
println("performance metrics relative path: ", benchmark_performance_metrics_relative_path);


instance_files = readdir(benchmark_instances_relative_path);
submission_files = readdir(benchmark_performer_submissions_relative_path);
performance_metrics_files = readdir(benchmark_performance_metrics_relative_path);




# Build the list of solution_uuids that have already been processed.
processed_solution_uuids = []
for performance_metric_file in performance_metrics_files
    fname = benchmark_performance_metrics_relative_path * performance_metric_file;
    if split(fname,".")[end] != "json"
        # skip over README.md and other supplementary files.
        continue
    end

    processed_solution_dict = JSON.parsefile(fname)
    solution_uuid = processed_solution_dict["solution_uuid"]
    if (this_script_sha1sum != processed_solution_dict["processed_with_perf_calculator_hash"])
        println("the file solution_uuid:$solution_uuid was already processed and the results are in $fname.  BUT the results in $fname were calculated with a different version of the performance calculator script.  The file $fname will be deleted.  The solution should be re-processed with the current version of the script.") #TODO: guarantee that we process it below!!

        rm(fname);
    else
        if (processed_solution_dict["solution_uuid"] in processed_solution_uuids)
            println("solution_uuid:$solution_uuid has already been processed. Skipping.");
        else
            push!(processed_solution_uuids, processed_solution_dict["solution_uuid"]);
        end
    end
end
    












for submission_file in submission_files
    fname = benchmark_performer_submissions_relative_path * submission_file
    if split(fname,".")[end] != "h5"
        # skip over README.md and other supplementary files.
        continue
    end

    solution = HDF5.h5read(fname, "solution");
    solution_uuid = solution["solution_uuid"];
    if !(solution_uuid in processed_solution_uuids)
        println("\n\nprocessing solution_uuid:$solution_uuid ... ");

        benchmark_instance_uuid = solution["instance_uuid"];
        performance_metrics_uuid = string(uuid4());
        
        performance_metrics_dict = Dict();
        performance_metrics_dict["solution_uuid"] = solution_uuid;
        performance_metrics_dict["instance_uuid"] = benchmark_instance_uuid;     
        performance_metrics_dict["performance_metrics_uuid"] = performance_metrics_uuid;
        performance_metrics_dict["processed_with_perf_calculator_hash"] = this_script_sha1sum;
        performance_metrics_dict["timestamp"] = string(now(UTC));



        # locate and open corresponding benchmark instance 
        located_corresponding_benchmark_instance = false;
        instance_fname = "";
        instance_data = Dict();
        for instance_file in instance_files
            instance_fname = benchmark_instances_relative_path * instance_file;
            if split(instance_fname,".")[end] != "json"
                # skip over README.md and other supplementary files.
                continue
            end

            instance_data = JSON.parsefile(instance_fname)
            if benchmark_instance_uuid == instance_data["metadata"]["instance_uuid"]
                located_corresponding_benchmark_instance = true;
                break
            end
        end
        @assert located_corresponding_benchmark_instance

        # extract metadata and parameters from original benchmark instance
        n = length(instance_data["graph_data"]["nodes"]);
        println("graph name: ", instance_data["metadata"]["graph_name"]);        
        println("number of nodes: $n");
        time_limit_seconds = instance_data["benchmark_requirements"]["time_limit_seconds"];
        num_samples_k = instance_data["benchmark_requirements"]["num_samples_k"];
        k_B = instance_data["metadata"]["k_B"];
        temperature_T = instance_data["metadata"]["temperature_T"];
        
        # reassemble original interaction_strength_J matrix from dictionary:
        interaction_strength_J = zeros(n,n);
        for edge in instance_data["graph_data"]["links"]
            i = edge["source"] + 1; # plus 1 because we are going from base0 to base1
            j = edge["target"] + 1;
            w = edge["weight"];
            interaction_strength_J[i,j] = w;
            interaction_strength_J[j,i] = w;
        end
        external_field_B = zeros(n);
        for node in instance_data["graph_data"]["nodes"]
            i = node["id"] + 1 # plus 1 because we are going from base0 to base1
            external_field_B[i] = node["B"];
        end
        # manually factor in the Boltzmann constant and temperature_T
        beta = (1/(k_B*temperature_T));
        interaction_strength_J *= beta;
        external_field_B *= beta;


        # collate the "samples" submitted by the benchmark performer
        counts = solution["solution_reported"]["states_observed_counts"];
        states = solution["solution_reported"]["states_observed"];
        # first column is number of observations (counts).  
        # we organize the samples in this way for compatibility with the GraphicalModelLearning package.
        samples = hcat(counts, transpose(states));

        # Use GraphicalModelLearning to estimate the parameters from the samples
        try
            GML_method = GraphicalModelLearning.logRISE();
            learned_adj_matrix = GraphicalModelLearning.learn(samples, GML_method);
            learned_external_field_B = [learned_adj_matrix[i,i] for i in 1:n];

            # vectorize all parameters J_ij, B_i to calculate differences
            learned_param_vector = [];
            original_param_vector = [];
            for i in 1:n
                for j in i+1:n 
                    push!(learned_param_vector, learned_adj_matrix[i,j]);
                    push!(original_param_vector, interaction_strength_J[i,j]);
                end
            end
            # a single parameter vector includes all J_ij and B_i terms.
            for i in 1:n
                push!(learned_param_vector, learned_external_field_B[i]);
                push!(original_param_vector, external_field_B[i]);
            end

            # parameter comparison norm metrics
            x = learned_param_vector - original_param_vector;
            tolerance = 1e-9;
            ell_0_norm = sum( abs.(x) .>= tolerance*ones(length(learned_param_vector)));
            ell_1_norm = sum( abs.(x) );
            ell_2_norm = sqrt( sum( x.^2 ) );
            ell_inf_norm = maximum( abs.(x) );
            performance_metrics_dict["ell_0_norm"] = ell_0_norm;
            performance_metrics_dict["ell_1_norm"] = ell_1_norm;
            performance_metrics_dict["ell_2_norm"] = ell_2_norm;
            performance_metrics_dict["ell_inf_norm"] = ell_inf_norm;
        catch ex
            println("Error occurred: $ex")
            performance_metrics_dict["ell_0_norm"] = "error_during_GML";
            performance_metrics_dict["ell_1_norm"] = "error_during_GML";
            performance_metrics_dict["ell_2_norm"] = "error_during_GML";
            performance_metrics_dict["ell_inf_norm"] = "error_during_GML";
        end                     


        # other probability-distribution-related metrics:
        if n > 25 # 2^25 seems to be the limit for the machine we run this script on.
            performance_metrics_dict["kl_divergence"] = "n_too_large_to_compute";
            performance_metrics_dict["total_variation"] = "n_too_large_to_compute";
        else
            J = interaction_strength_J
            for i in 1:n
                for j in 1:i
                    J[i,j] = 0; # set this to an upper triangular matrix with zero diagonal.
                end
            end
            
            
            true_pdf = zeros(2^n)
            for state_index in 0:2^n-1
                spin_config = convert_state_index_to_spin_config(state_index, n);
                energy = transpose(spin_config)*J*spin_config + transpose(external_field_B)*spin_config;
                true_pdf[state_index + 1] = exp(energy); # note that -beta is already factored into J,B
            end
            Z = sum(true_pdf);
            true_pdf *= (1/Z);

            
            empirical_pdf = zeros(2^n);
            num_unique_states_observed = size(samples)[1];
            for sample_row in 1:num_unique_states_observed
                count = samples[sample_row,1];
                spin_config = samples[sample_row,2:end];
                # println(count , spin_config)
                state_index = convert_spin_config_to_state_index(spin_config);
                empirical_pdf[state_index + 1] = count;
            end
            empirical_pdf *= 1.0/sum(empirical_pdf);
            
            
            kl_divergence = 0.0;
            for state_index = 0:2^n-1
                p = empirical_pdf[state_index + 1];
                if p < 1e-9 # p is zero or very close 
                    # the contribution to kl_divergence is zero because lim_{x->0} xlogx = 0.
                    # TODO: figure out a way to vectorize this math instead of a loop
                    # but you have to fix up the log(0)=-Inf issue.
                    continue 
                end
                
                q = true_pdf[state_index + 1];
                kl_divergence += p*log(p/q);
            end

            total_variation_distance = maximum( abs.(empirical_pdf - true_pdf));

            performance_metrics_dict["kl_divergence"] = kl_divergence;
            performance_metrics_dict["total_variation"] = total_variation_distance;
        end




        # runtime metrics (average sample time)
        overall_time_seconds = solution["run_time"]["overall_time"]["seconds"];
        counts = solution["solution_reported"]["states_observed_counts"];
        num_samples_returned = sum(counts);
        performance_metrics_dict["returned_required_num_samples_k"] = (num_samples_returned >= num_samples_k);
        performance_metrics_dict["within_time_limit"] = (overall_time_seconds <= time_limit_seconds);     
        performance_metrics_dict["overall_time"] = overall_time_seconds;
        performance_metrics_dict["average_seconds_per_sample"] = overall_time_seconds/num_samples_returned;



 


        # write out dictionary to JSON file
        output_filename = benchmark_performance_metrics_relative_path * "performance_metrics." * performance_metrics_uuid * ".json";
        open(output_filename, "w") do io
            JSON.print(io, performance_metrics_dict);
        end
    end
end

println("done");

