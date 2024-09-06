import torch
import torchvision
import json
import argparse


from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import functional as F


# 设置命令行参数解析
parser = argparse.ArgumentParser(description="CIFAR10 Dataset Partitioning")
parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
parser.add_argument('--balance', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to balance the data')

# 解析命令行参数
args = parser.parse_args()

# 从命令行获取的参数
num_clients = args.num_clients
balance = args.balance

trainset = torchvision.datasets.CIFAR10(root="/ailab/user/gushuang/data", train=True, download=True)
testset = torchvision.datasets.CIFAR10(root="/ailab/user/gushuang/data", train=False, download=True)


num_classes = 10

seed = 2021


trainset_part = CIFAR10Partitioner(trainset.targets, 
                                num_clients,
                                balance=balance, 
                                partition="iid",
                                unbalance_sgm=0.6,
                                seed=seed)
# save to pkl file
balance_str = 'balance' if balance else 'unbalance'
filename = f"train_{num_clients}clients_{balance_str}_iid.pkl"
torch.save(trainset_part.client_dict, filename)

testset_part = CIFAR10Partitioner(testset.targets, 
                                num_clients,
                                balance=True, 
                                partition="iid",
                                unbalance_sgm=0.3,
                                seed=seed)
# save to pkl file
balance_str = 'balance' if balance else 'unbalance'
filename = f"test_{num_clients}clients_{balance_str}_iid.pkl"
torch.save(testset_part.client_dict, filename)


def calculate_weights():
    weights = {}
    for i, indices in trainset_part.client_dict.items():
        weights[f'site-{i + 1}'] = len(indices)/50000.0
     
    return weights

with open('jobs/app/config/config_staleness.json', 'r') as file:
    config = json.load(file)


config['weights'] = calculate_weights()
print(config['weights'])

with open('jobs/app/config/config_staleness.json', 'w') as file:
    json.dump(config, file, indent=2)