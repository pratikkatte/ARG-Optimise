import pickle
import math

import torch


def build_cosine_lr_scheduler(optimizer, total_steps, min_factor=0.1):
    total_steps = max(int(total_steps), 1)
    min_factor = float(min_factor)
    if min_factor < 0.0 or min_factor > 1.0:
        raise ValueError("min_factor must be between 0.0 and 1.0")

    def lr_lambda(step):
        progress = min(max(float(step) / total_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
def read_fasta(filepath):
    all_seqs_dict = {}
    with open(filepath, 'r') as file:
        seq_id = None
        all_seqs = []
        for line in file:
            line = line.rstrip()
            if line.startswith('>'):
                if len(all_seqs) > 0 and seq_id is not None:
                    all_seqs_dict[seq_id] = all_seqs
                seq_id = line
                all_seqs = []
            elif len(line) > 0:
                all_seqs.append(line)

        if len(all_seqs) > 0 and seq_id is not None:
            all_seqs_dict[seq_id] = all_seqs

    return all_seqs_dict

def load_sequences(sequences_path):
    if sequences_path.endswith('.fa'):
        key_to_seqs_dict = read_fasta(sequences_path)
        all_seqs = ["".join(lines) for lines in key_to_seqs_dict.values()]
    elif sequences_path.endswith('.pickle'):
        data = pickle.load(open(sequences_path, 'rb'))
        all_seqs = list(data.values()) if isinstance(data, dict) else data
    else:
        all_seqs = pickle.load(open(sequences_path, 'rb'))

    return [seq.replace('?', '-') for seq in all_seqs]
