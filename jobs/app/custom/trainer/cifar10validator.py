# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor,Resize
from torchvision import models
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class Cifar10Validator(Executor):
    def __init__(self, data_path="~/data", validate_task_name=AppConstants.TASK_VALIDATION):
        super().__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        # self.model = resnet18()
        # self.model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        # self.model = SimpleNetwork()
        # self.model = resnet18()
        # self.model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        # self.model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        self.model = models.resnet.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        in_fea = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_fea, 10) 
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        # Preparing the dataset for testing.
        transforms = Compose(
            [
                ToTensor(),
                Resize(224),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self._test_dataset = CIFAR10(root=data_path, train=False, transform=transforms)
        self.testdata = self.testloader()
    

    def testloader(self):
        with open('/ailab/user/gushuang/experiment/asyn/CIFAR10_fedasync/test_6clients_balance_iid.pkl', 'rb') as f:
            loaded_data = torch.load(f)


        client_loaders = {}

        for i, client_indices in loaded_data.items():
            client_dataset = Subset(self._test_dataset, client_indices)
            client_loader = DataLoader(client_dataset, batch_size=128, shuffle=True)
            client_loaders[f'site-{i+1}'] = client_loader

        return client_loaders

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self._validate(weights, fl_ctx, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, fl_ctx, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()

        test_site = fl_ctx.get_identity_name()
        test_loader = self.testdata[test_site]
        
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]

            metric = correct / float(total)

        return metric
