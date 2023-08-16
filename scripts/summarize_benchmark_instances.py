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


# add some relative paths so we can import sampling.py module.
import sys
sys.path.append("../")
sys.path.append("../src")
sys.path.append("./src")
sys_path = sys.path

import json

import copy

import os

import csv




relative_path = "../benchmark_instances"
benchmark_instances = os.listdir(relative_path)


instance_list = []
for filename in benchmark_instances:

    if ".json" in filename:
        print("parsing", filename,"...")
        full_filename = relative_path + "/" + filename
        with open(full_filename,"rb") as f:
            read_bytes = f.read()
            f.close()

            # convert the byte array to a dict
            full_instance = json.loads(read_bytes)
            instance = copy.copy(full_instance["metadata"])
            
            # fixer-upper: include more fields...
            instance["filename"] = filename
            instance["num_samples_k"] = full_instance["benchmark_requirements"]["num_samples_k"]
            instance["time_limit_seconds"] = full_instance["benchmark_requirements"]["time_limit_seconds"]

            instance_list.append(instance)





fieldnames = list(instance.keys())

with open("../benchmark_instances/benchmark_instance_summary.csv", mode="w") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    for instance in instance_list:
        writer.writerow(instance)



print("done")
