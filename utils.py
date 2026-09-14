import numbers
import pickle

import torch

from env.actions import CoalescenceChoice, RecombinationChoice

def build_scheduler(optimizer, cfg_scheduler):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)
    return scheduler

    
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


def action_as_dict(action):
    """Encode an action using the existing replay-record format."""
    if isinstance(action, dict):
        return dict(action)
    if isinstance(action, CoalescenceChoice):
        record = dict(event_type="coal", active_lineage_i=int(action.active_lineage_i),
                      active_lineage_j=int(action.active_lineage_j))
    elif isinstance(action, RecombinationChoice):
        record = dict(event_type="recomb", active_lineage_i=int(action.active_lineage_i),
                      breakpoint=int(action.breakpoint) if action.breakpoint is not None else None,
                      material_count=int(action.material_count), span_start=int(action.span_start),
                      span_end=int(action.span_end))
    else:
        raise ValueError(f"Unknown ARG action: {action}")
    if action.time_action is not None:
        record["time_action"] = int(action.time_action)
    if action.delta_t is not None:
        record["delta_t"] = float(action.delta_t)
    return record


def action_from_dict(record):
    """Decode a saved replay record into its action dataclass."""
    if not isinstance(record, dict):
        raise ValueError("Expected an ARG action record")
    kind = record.get("event_type")
    if kind == "coal":
        cls, fields = CoalescenceChoice, ("active_lineage_i", "active_lineage_j")
    elif kind == "recomb":
        cls, fields = RecombinationChoice, ("active_lineage_i", "material_count", "span_start", "span_end")
    else:
        raise ValueError(f"Unknown ARG action event_type: {kind}")
    values = {}
    for name in fields:
        if not isinstance(record.get(name), numbers.Integral):
            raise ValueError(f"Invalid ARG action field: {name}")
        values[name] = int(record[name])
    for name in (("time_action", "breakpoint") if kind == "recomb" else ("time_action",)):
        value = record.get(name)
        if value is not None:
            if not isinstance(value, numbers.Integral):
                raise ValueError(f"Invalid ARG action field: {name}")
            values[name] = int(value)
    if record.get("delta_t") is not None:
        values["delta_t"] = float(record["delta_t"])
    return cls(**values)
