import torch

import json
#with open('config.json') as f:
#    config = json.load(f)

#reconstruction_target = config['reconstruction_target']
reconstruction_target = 'one_hot_pid'

def pad_or_truncate(event, max_seq_length=256, total_charge_index=int(16)):
    """
    Pad or truncate an event to the given max sequence length, and create an attention mask.
    
    Args:
    - event: Tensor of shape (seq_length, feature_dim) where seq_length can vary.
    - max_seq_length: Maximum sequence length to pad/truncate to.
    
    Returns:
    - Padded or truncated event of shape (max_seq_length, feature_dim).
    - Attention mask of shape (max_seq_length) where 1 indicates a valid token and 0 indicates padding.
    """
    seq_length = event.size(0)
    
    # Truncate if the sequence is too long
    if seq_length > max_seq_length:
        # sort the event by total charge
        event = event[event[:, total_charge_index].argsort(descending=True)]
        truncated_event = event[:max_seq_length]
        return truncated_event, max_seq_length
    

    # Pad if the sequence is too short
    elif seq_length < max_seq_length:
        padding = torch.zeros((max_seq_length - seq_length, event.size(1)))
        padded_event = torch.cat([event, padding], dim=0)
        return padded_event,  seq_length
    
    # No need to pad or truncate if it's already the correct length
    return event, seq_length

def custom_collate_fn(batch, max_seq_length=256):
    """
    Custom collate function to pad or truncate each event in the batch.
    
    Args:
    - batch: List of (event, label) tuples where event has a variable length [seq_length, 7].
    - max_seq_length: The fixed length to pad/truncate each event to (default is 512).
    
    Returns:
    - A batch of padded/truncated events with shape [batch_size, max_seq_length, 7].
    - Corresponding labels.
    """
    # Separate events and labels
    events = [item.x for item in batch]  # Each event has shape [seq_length, 7]
    
    padded_events, event_lengths = zip(*[pad_or_truncate(event, max_seq_length) for event in events])

    batch_events = torch.stack(padded_events)
    event_lengths = torch.tensor(event_lengths)

    # Extract labels and convert to tensors
    label_name = reconstruction_target

    # Extract labels and convert to tensors (3D vectors)
    vectors = [item[label_name] for item in batch]

    # Stack labels in case of multi-dimensional output
    labels = torch.stack(vectors)

    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    
    return batch_events, labels, event_lengths
