import torch


def _has_mps():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(requested_device):
    device = (requested_device or "auto").strip().lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _has_mps():
            return "mps"
        return "cpu"

    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return requested_device
        if _has_mps():
            print(f"Requested device '{requested_device}' is unavailable; falling back to mps.")
            return "mps"
        print(f"Requested device '{requested_device}' is unavailable; falling back to cpu.")
        return "cpu"

    if device == "mps":
        if _has_mps():
            return "mps"
        if torch.cuda.is_available():
            print("Requested device 'mps' is unavailable; falling back to cuda.")
            return "cuda"
        print("Requested device 'mps' is unavailable; falling back to cpu.")
        return "cpu"

    if device == "cpu":
        return "cpu"

    return requested_device