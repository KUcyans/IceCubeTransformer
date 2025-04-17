import torch

def print_info():
    print("Torch file:", torch.__file__)
    print("Torch version:", torch.__version__)
    print("CUDA built:", torch.backends.cuda.is_built())
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())

def lock_and_load():
    """Set CUDA device based on config['gpu'] if available, else use CPU."""
    print("torch.cuda.is_available():", torch.cuda.is_available())
    available_devices = list(range(torch.cuda.device_count()))
    print(f"Available CUDA devices: {available_devices}")

    if torch.cuda.is_available():
        selected_gpu = available_devices[0]  # Default to the first available GPU
        torch.cuda.empty_cache()
        print("🔥 LOCK AND LOAD! GPU ENGAGED! 🔥")
        device = torch.device(f"cuda:{selected_gpu}")  # ✅ Use the correct index
        torch.cuda.set_device(selected_gpu)  # ✅ Explicitly set device
        torch.set_float32_matmul_precision('highest')
        print(f"Using GPU: {selected_gpu} (cuda:{selected_gpu})")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    print(f"Selected device: {device}")
    return device

def run(device):
    query = torch.randn(2, 4, 8, device=device)
    key = torch.randn(2, 4, 8, device=device)
    value = torch.randn(2, 4, 8, device=device)
    attn_mask = None
    dropout_p = 0.0
    is_causal = False

    # Call scaled_dot_product_attention to trigger your C++ backend logging
    output = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal)

    print(output)

def main():
    print_info()
    device = lock_and_load()
    run(device)
if __name__ == "__main__":
    main()
    
    
    

