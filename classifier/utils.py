from collections import defaultdict
import json
import torch

class GpuMem:
    def __init__(self, device=0):
        self.device = device
        self.peak = None

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.before = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize(self.device)
        self.peak = torch.cuda.max_memory_allocated(self.device)
    
    def memory_usage(self):
        '''get memory usage (measured in GB)'''
        unit = 1e9
        return (self.peak - self.before) / unit

def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def merge_dicts_of_lists(dataset_list) -> dict:
    """
    将一系列 dict(键→list) 合并为一个 dict，
    同一个键对应的 list 会被 extend 到一起。
    """
    merged = defaultdict(list)
    for d in dataset_list:
        for key, value in d.items():
            # 如果 value 本身是 list，则 extend；否则 append
            if isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return dict(merged)

def load_training_data(train_dataset_list):
    dataset_list = []
    for data_name in train_dataset_list:
        dataset = load_data(data_name)
        dataset_list.append(dataset)
    ## combine training data
    train_data = merge_dicts_of_lists(dataset_list)
    return train_data

def load_rewrite_data(rewrite_data_list):
    dataset_list = []
    for data_name in rewrite_data_list:
        data_file = f"{data_name}.json"
        with open(data_file, "r") as fin:
            dataset = json.load(fin)
            print(f"Raw rewrite data loaded from {data_file}")
        dataset = {
            'rewrite_original': [x['rewrite_original'] for x in dataset], 
            'rewrite_sampled': [x['rewrite_sampled'] for x in dataset]
        }
        dataset_list.append(dataset)
    train_rewrite_data = merge_dicts_of_lists(dataset_list)
    return train_rewrite_data

def load_training_data2(train_dataset_list, base_dir):
    dataset_list = []
    for data_name in train_dataset_list:
        data_file = f'{base_dir}/{data_name}'
        with open(data_file, "r") as fin:
            data = json.load(fin)
            print(f"Raw data loaded from {data_file}")
        dataset_list.append(data)
    ## combine training data
    train_data = merge_dicts_of_lists(dataset_list)
    return train_data

def separated_string(s: str):
    '''
    return a list of strings from a string
    '''
    return s.split('&')
