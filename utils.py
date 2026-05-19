import pickle

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
        all_seqs = list(key_to_seqs_dict.values())[0]
    elif sequences_path.endswith('.pickle'):
        data = pickle.load(open(sequences_path, 'rb'))
        all_seqs = list(data.values()) if isinstance(data, dict) else data
    else:
        all_seqs = pickle.load(open(sequences_path, 'rb'))

    return [seq.replace('?', '-') for seq in all_seqs]
