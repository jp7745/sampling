#!/usr/bin/env python3



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


import os
import json
import datetime
import uuid
import platform

import h5py


import numpy as np

# add some relative paths so we can import sampling.py module.
import sys
sys.path.append("../")
sys.path.append("../src")
sys.path.append("./src")
sys_path = sys.path

import sampling








def create_hdf5_structure(parent_group: h5py.Group, python_dict: dict):
    """A utility function to Recursively create the HDF5 file structure from a Python dictionary.
    """
    # TODO: implement compression
    # compression_options = {"compression":"gzip", "compression_opts":9}

    for key, val in python_dict.items():
        if isinstance(val, dict):
            subgroup = parent_group.create_group(key)
            create_hdf5_structure(subgroup, val)
        elif isinstance(val, list):
            if isinstance(val[0], dict):
                # handle a list of dictionaries
                # TODO: fix this hacky way of consolidating dict/json/hdf5
                nested_group = parent_group.create_group(key)
                for j, nested_dict in enumerate(val):
                    nested_subgroup = nested_group.create_group(f'_list_item_{j}_')
                    create_hdf5_structure(nested_subgroup, nested_dict)
            else:
                dataset = parent_group.create_dataset(
                        key,
                        data=val
                    )
        else:
            if val is None:
                val = "_null_" #TODO: fix this hacky way of consolidating dict/json/hdf5
            parent_group.create_dataset(
                    key,
                    data=val
                )
















instances_relative_path = "../benchmark_instances"
benchmark_instances = os.listdir(instances_relative_path)


solution_relative_path = "../benchmark_performer_submissions"



for filename in benchmark_instances:

    if ".json" in filename:
        print("running", filename,"...")

        full_filename = instances_relative_path + "/" + filename
        
        with open(full_filename,"rb") as f:
            read_bytes = f.read()
            f.close()
        # convert the byte array to a dict
        instance_dict = json.loads(read_bytes)

        num_samples_k = instance_dict["benchmark_requirements"]["num_samples_k"]
        instance_uuid = instance_dict["metadata"]["instance_uuid"]

        overall_start_time = datetime.datetime.utcnow()

        preprocessing_start_time = datetime.datetime.utcnow()
        G = sampling.load_graph_instance_from_file(full_filename)
        preprocessing_stop_time = datetime.datetime.utcnow()
        preprocessing_time = preprocessing_stop_time - preprocessing_start_time
        preprocessing_seconds = preprocessing_time.total_seconds()        

        n = len(G.nodes())


        algorithm_start_time = datetime.datetime.utcnow()
        
        
        # samples are initially stored in a growing dictionary.  
        # pre-allocating a vector to count all possible states is not feasible (too many states)
        samples = {}

        for sample in range(num_samples_k):

            # print status:
            if sample % 50 == 0:
                print("sample:",sample,"/",num_samples_k,"...")  


            # random starting spin_config:
            spin_config = sampling.generate_random_spin_config(n)
            G.set_spins(spin_config)

            # run MCMC to generate one sample:
            sample_energy, sample_config = G.generate_mcmc_sample_v1(num_steps=500)
            
            final_state_index = sampling.convert_spin_config_to_state_index(sample_config)

            try:
                samples[final_state_index] += 1 # increment counter of observances
            except:
                samples[final_state_index] = 1

        

        algorithm_stop_time = datetime.datetime.utcnow()
        algorithm_time = algorithm_stop_time - algorithm_start_time
        algorithm_seconds = algorithm_time.total_seconds()

        postprocessing_start_time = datetime.datetime.utcnow()
        # postprocessing steps here.  (data reading, measuring, etc. if necessary.)
        postprocessing_stop_time = datetime.datetime.utcnow()
        postprocessing_time = postprocessing_stop_time - postprocessing_start_time
        postprocessing_seconds = postprocessing_time.total_seconds()

        overall_stop_time = datetime.datetime.utcnow()
        overall_time = overall_stop_time - overall_start_time
        overall_seconds = overall_time.total_seconds()



        run_time = {}
        run_time["overall_time"] = {}
        run_time["preprocessing_time"] = {}
        run_time["algorithm_run_time"] = {}
        run_time["postprocessing_time"] = {}

        run_time["overall_time"]["wall_clock_start_time"] = str(overall_start_time)
        run_time["overall_time"]["wall_clock_stop_time"] = str(overall_stop_time)
        run_time["overall_time"]["seconds"] = overall_seconds

        run_time["preprocessing_time"]["wall_clock_start_time"] = str(preprocessing_start_time)
        run_time["preprocessing_time"]["wall_clock_stop_time"] = str(preprocessing_stop_time)
        run_time["preprocessing_time"]["seconds"] = preprocessing_seconds

        run_time["algorithm_run_time"]["wall_clock_start_time"] = str(algorithm_start_time)
        run_time["algorithm_run_time"]["wall_clock_stop_time"] = str(algorithm_stop_time)
        run_time["algorithm_run_time"]["seconds"] = algorithm_seconds

        run_time["postprocessing_time"]["wall_clock_start_time"] = str(postprocessing_start_time)
        run_time["postprocessing_time"]["wall_clock_stop_time"] = str(postprocessing_stop_time)
        run_time["postprocessing_time"]["seconds"] = postprocessing_seconds






        # convert dictionary into a histogram-like numpy arrays.
        num_unique_states_observed = len(samples)
        states_observed = np.zeros((num_unique_states_observed, n))
        states_observed_counts = np.zeros(num_unique_states_observed)
        i = 0
        for key, val in samples.items():
            spin_config = sampling.convert_state_index_to_spin_config(n, key)
            states_observed[i,:] = spin_config
            states_observed_counts[i] = val
            i += 1








        # create solution dictionary
        # TODO: automate more of the metadata associated with this.
        solution_uuid = str(uuid.uuid4())
        solution_filename = solution_relative_path + "/TESTING.solution." + solution_uuid + ".h5"

        benchmark_performer_solution = {
            "solution":{
                "solution_timestamp": str(datetime.datetime.utcnow().isoformat()),
                "instance_uuid":instance_uuid,
                "solution_uuid":solution_uuid,
                "benchmark_performer_contact_info":[
                    {
                        "name": "BOBQAT Team",
                        "email": "bobqat@l3harris.com",
                        "institution": "L3Harris Technologies, Inc.",
                        "website":"www.l3harris.com"
                    }
                ],
                "compute_details":{
                    "hardware_type":"classical_computer",
                    "hardware_details":{
                        "name": "VLE",
                        "compute_node_specifications":[
                            {
                                "name":platform.node(),
                                "total_memory_gb": 16,
                                "memory_details": "4GB",
                                "total_cpus": 2,
                                "cpu_details":"Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz",
                                "total_disk": None,
                                "disk_details":None,
                                "gpu_details":None
                            }
                        ]
                    },
                    "software_details":{
                        "name": "MCMC-MH Sampler v1.0",
                        "version": "1.0",
                        "references":None,
                        "non_default_parameters":None,
                        "python_version":platform.python_version(),
                        "host_os":str(platform.uname())
                    }
                },
                "run_time":run_time, # as calculated above.
                "compute_resources_used":None, # memory, CPU, etc.
                "solution_reported":{
                    "states_observed":states_observed, # TODO: convert these to integers for better storage. Reconcile format with GraphicalModelLearning.jl format.
                    "states_observed_counts":states_observed_counts, 
                }
            },
            "digital_signature":{
                "description":"calculated over the \"solution\" object.  Not implemented at this time."
            }
        }


        # write solution dictionary to HDF5 file
        with h5py.File(solution_filename, "w") as f:
            create_hdf5_structure(f, benchmark_performer_solution)
            f.close()







print("done")
