# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import threading
from typing import Optional
from collections import deque
import os
import json


class AsyncAggregationHelper(object):
    def __init__(self, exclude_vars: Optional[str] = None, weigh_by_local_iter: bool = True):
        """Perform weighted aggregation.

        Args:
            exclude_vars (str, optional): regex string to match excluded vars during aggregation. Defaults to None.
            weigh_by_local_iter (bool, optional): Whether to weight the contributions by the number of iterations
                performed in local training in the current round. Defaults to `True`.
                Setting it to `False` can be useful in applications such as homomorphic encryption to reduce
                the number of computations on encrypted ciphertext.
                The aggregated sum will still be divided by the provided weights and `aggregation_weights` for the
                resulting weighted sum to be valid.
        """
        super().__init__()
        self.lock = threading.Lock()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.weigh_by_local_iter = weigh_by_local_iter
        self.reset_stats()
        self.weight_list = deque()
        self.history = deque()

    def reset_stats(self):
        self.history = deque()

    def add(self, data, weight, contributor_name, contribution_round, site_round):
        """Compute weighted sum and sum of weights."""
        with self.lock:
            for k, v in data.items():
                if self.exclude_vars is not None and self.exclude_vars.search(k):
                    continue
                if self.weigh_by_local_iter:
                    data[k] = v * weight
                else:
                    data[k] = v
            self.weight_list.append(data)  
                  
        self.history.append(
            {
                "contributor_name": contributor_name,
                "round": contribution_round,
                "site_round": site_round,
                "weight": weight,
            }
        )

        
    def get_result(self, fl_ctx, global_weight,current_round,config_staleness_filename):
        """Divide weighted sum by sum of weights."""
        aggregated_dict={}
        with self.lock:
            current_aggregate_weight = self.weight_list.popleft()
            current_aggregate_client = self.history.popleft()
            # site_round = current_aggregate_client["site_round"]

            alpha_t = self.func_s(fl_ctx=fl_ctx, current_round=current_round, aggregate_client=current_aggregate_client, config_staleness_filename=config_staleness_filename)
            print(alpha_t)
            for key, local in current_aggregate_weight.items():
                if key in global_weight:
                    global_w = global_weight[key]
                    avg_value = (alpha_t*local + (1-alpha_t)*global_w)
                    aggregated_dict[key] = avg_value
                else:
                    aggregated_dict[key] = local
    
            for key, global_w in global_weight.items():
                if key not in current_aggregate_weight:
                    aggregated_dict[key] = global_w

            return current_aggregate_client,aggregated_dict       

    def func_s(self, fl_ctx, current_round, aggregate_client, config_staleness_filename):
        config_path = fl_ctx.get_engine().get_workspace().get_app_config_dir(fl_ctx.get_job_id())
        print(config_path)
        staleness_config_path = os.path.join(config_path, config_staleness_filename)
      
        with open(staleness_config_path) as file:
            config_info = json.load(file)
        
        site_round = aggregate_client["site_round"]
        site_name = aggregate_client["contributor_name"]
        staleness = config_info["staleness"]
        alpha = config_info["alpha"]
        a = config_info["a"]
        b = config_info["b"]
        if staleness=="const":
            return alpha*1
        if staleness=="data_weighted":
            return config_info["weights"].get(site_name)
        if staleness=="poly":
            return alpha/pow(current_round-site_round + 1,a)
        if staleness=="hinge":
            if current_round-site_round > b:
                return alpha/(float(current_round - site_round - b)*a + 1)
            else:
                return alpha*1

        
    def get_history(self):
        return self.history

    def get_len(self):
        return len(self.get_history())
