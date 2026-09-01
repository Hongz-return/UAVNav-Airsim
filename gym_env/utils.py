"""Shared utilities for building Mamba/LSTM/Transformer policy kwargs from config."""

import ast
from configparser import ConfigParser


def build_mamba_policy_kwargs(cfg: ConfigParser) -> dict:
    """
    Build ``policy_kwargs`` dict for MambaSACPolicy from a ConfigParser.

    This centralises the parameter extraction that would otherwise be
    duplicated across the training entrypoint (both new-model and resume
    branches) and the evaluation entrypoint.

    Args:
        cfg: ConfigParser with ``[options]`` and ``[mamba]`` sections.

    Returns:
        dict ready to be passed as ``policy_kwargs`` to MambaSAC.
    """
    force_lstm = cfg.getboolean('options', 'use_lstm_baseline', fallback=False)
    mamba_backend = cfg.get('mamba', 'mamba_backend', fallback='auto')
    if mamba_backend == 'lstm':
        force_lstm = True

    return {
        'cnn_output_dim': cfg.getint('mamba', 'cnn_output_dim', fallback=64),
        'cnn_attention': cfg.get('mamba', 'cnn_attention', fallback='none'),
        'cnn_attention_reduction': cfg.getint('mamba', 'cnn_attention_reduction', fallback=8),
        'cnn_attention_stage': cfg.get('mamba', 'cnn_attention_stage', fallback='auto'),
        'mamba_hidden_dim': cfg.getint('mamba', 'mamba_hidden_dim', fallback=128),
        'mamba_d_state': cfg.getint('mamba', 'mamba_d_state', fallback=16),
        'mamba_d_conv': cfg.getint('mamba', 'mamba_d_conv', fallback=4),
        'mamba_expand': cfg.getint('mamba', 'mamba_expand', fallback=2),
        'actor_hidden': ast.literal_eval(cfg.get('mamba', 'actor_hidden', fallback='[128, 64]')),
        'critic_hidden': ast.literal_eval(cfg.get('mamba', 'critic_hidden', fallback='[128, 128]')),
        'force_lstm': force_lstm,
        'inference_seq_len': cfg.getint('mamba', 'inference_seq_len', fallback=8),
        'cnn_update_mode': cfg.get('mamba', 'cnn_update_mode', fallback='both'),
        'backend': mamba_backend,
        'transformer_heads': cfg.getint('mamba', 'transformer_heads', fallback=4),
        'transformer_layers': cfg.getint('mamba', 'transformer_layers', fallback=2),
    }


def get_backend_display_name(policy_kwargs: dict) -> str:
    """Return a human-readable backend name for logging."""
    backend = policy_kwargs.get('backend', 'auto')
    if backend == 'transformer':
        heads = policy_kwargs.get('transformer_heads', 4)
        layers = policy_kwargs.get('transformer_layers', 2)
        return f"Transformer(heads={heads}, layers={layers})"
    if policy_kwargs.get('force_lstm', False) or backend == 'lstm':
        return "LSTM"
    return f"Mamba({backend})"
