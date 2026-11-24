import torch
from transformers import AutoProcessor, AutoModelForVision2Seq 

import random
import numpy as np
import os

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_vlm_model(model_name, device_map="cuda"):
    '''
    Load a Vision-Language Model (VLM) and its processor, and freeze all parameters.

    This utility function:
        - Loads a pretrained Qwen2-VL (or compatible) Vision2Seq model. 
        - Freezes all model parameters to prevent any gradient update. 

    Args:
        model_name (str):
            Name or path of the Hugging Face VLM checkpoint.
            Example: 'Qwen/Qwen2-VL-7B-Instruct'
        
        device_map (str or dict):
            Device placement strategy. Defaults to "cuda".

    Returns:
        model (PreTrainedModel):
            Loaded Vision2Seq model with all parameters set to requires_grad=False.
        processor (AutoProcessor):
            Corresponding processor that handles multimodal tokenization, image transforms,
            and prepares inputs for the model.
    '''

    # Load the pretrained Vision-Language model and place its weights across devices.
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map=device_map
    )

    # Freeze all model parameters  
    for param in model.parameters():
        param.requires_grad = False

    # Load the corresponding multimodal processor for text/image encoding
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor

def register_embedding_gradient_mask(model, prefix_token_ids, num_special_tokens_in_prefix):
    """
    Register a gradient hook so that only the prefix-token embeddings receive updates.
    """
    # Ensure full embedding matrix is marked as trainable
    model.get_input_embeddings().weight.requires_grad = True

    # Indices of embeddings that should receive gradients
    embeddings_to_update = torch.tensor(
        prefix_token_ids[:num_special_tokens_in_prefix],
        dtype=torch.long
    ).to(model.device)

    def grad_hook(grad):
        # Zero out gradients for all tokens except prefix tokens
        mask = torch.zeros_like(grad)
        mask[embeddings_to_update] = 1.0
        return grad * mask

    # Install the backward hook on the embedding weight matrix
    hook_handle = model.get_input_embeddings().weight.register_hook(grad_hook)

    return hook_handle

def setup_prefix_tokens_and_mask(spmodel, processor, train_config):
    """
    Add trainable prefix soft tokens and register a gradient mask so that
    only prefix token embeddings receive gradients in Stage 2.
    """
    num_prefix = train_config["num_special_tokens_in_prefix"]

    # Create tokens: <|reserved_special_token_0|>, <|reserved_special_token_1|>, ...
    prefix_token_strs = [
        f"<|reserved_special_token_{i}|>" for i in range(num_prefix)
    ]

    # Add new tokens to tokenizer and expand LM embeddings
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": prefix_token_strs}
    )
    spmodel.vlm.resize_token_embeddings(len(processor.tokenizer))

    # Convert token strings → ids
    prefix_token_ids = processor.tokenizer.convert_tokens_to_ids(
        prefix_token_strs
    )

    # Combine into one long prefix string to prepend to input text
    prefix = "".join(prefix_token_strs)

    # Enable gradient updates only for prefix embeddings
    embedding = spmodel.vlm.get_input_embeddings()
    embedding.weight.requires_grad = True
    hook_handle = register_embedding_gradient_mask(
        spmodel.vlm,
        prefix_token_ids,
        num_prefix
    )

    return prefix, hook_handle

def save_soft_prompt_embeddings(model, processor, num_special_tokens_in_prefix, save_ckpt):
    """
    Save soft-prompt token embeddings into a checkpoint file.

    The function reconstructs reserved prefix token strings:
        <|reserved_special_token_0|>, ..., <|reserved_special_token_{N-1}|>
    and saves their embedding vectors as:
        {token_str: embedding_tensor(cpu)}.
    """
    if save_ckpt is None:
        return 

    # Reconstruct reserved prefix token strings
    prefix_token_strs = [
        f"<|reserved_special_token_{i}|>" for i in range(num_special_tokens_in_prefix)
    ]

    # Convert token strings to ids using the tokenizer
    prefix_token_ids = processor.tokenizer.convert_tokens_to_ids(prefix_token_strs)

    # Full embedding matrix
    embedding_weights = model.get_input_embeddings().weight.data

    saved_embeddings = {}

    # Extract embedding vector per prefix token id
    for token_str, token_id in zip(prefix_token_strs, prefix_token_ids):
        # Safety check: ensure the token is known by tokenizer
        if token_id == processor.tokenizer.unk_token_id:
            raise ValueError(f"{token_str} is not registered in the tokenizer.")
        saved_embeddings[token_str] = embedding_weights[token_id].cpu()

    # Write checkpoint to file
    torch.save(saved_embeddings, save_ckpt)

def save_router_fusion_ckpt(spmodel, save_path):
    """
    Save q_view, cross_attn, and fusion (if exists) into a checkpoint file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "q_view": {k: v.detach().cpu() for k, v in spmodel.q_view.items()},
        "cross_attn": {k: v.state_dict() for k, v in spmodel.cross_attn.items()},
        "fusion": spmodel.fusion.state_dict() if hasattr(spmodel, "fusion") else None,
        "router": spmodel.router.state_dict() if hasattr(spmodel, "router") else None
    }

    torch.save(ckpt, save_path)
    print(f"[✔] Saved checkpoint to {save_path}")
