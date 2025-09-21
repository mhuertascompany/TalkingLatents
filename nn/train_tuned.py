import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, Any

from .train import LLMTrainer


class LLMTrainerTuned(LLMTrainer):
    """
    LLMTrainer variant with:
    - Label smoothing on answer tokens
    - Token-level accuracy over answer tokens
    """
    def __init__(self, *args, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def get_loss(self, logits, target_ids, attention_mask=None):
        # Shift for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Only compute loss on actual answer tokens (not -100)
        active_mask = (shift_labels != -100)
        if not active_mask.any():
            return torch.tensor(0.0, device=logits.device), torch.tensor([0.0], device=logits.device)

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = active_mask.view(-1)

        active_logits = flat_logits[flat_mask]
        active_labels = flat_labels[flat_mask]

        # Label smoothing cross entropy
        loss = F.cross_entropy(active_logits, active_labels, reduction='mean', label_smoothing=self.label_smoothing)

        # Token-level accuracy on answer tokens
        with torch.no_grad():
            preds = active_logits.argmax(dim=-1)
            acc = (preds == active_labels).float().mean()
        return loss, acc.view(1)

    def train_batch(self, batch: Dict[str, Any], batch_idx: int, device: torch.device):
        input_ids = batch['input_ids'].to(device)
        input_spectra = batch['masked_spectra'].to(device)
        special_token_positions = batch['feature_start_indices'].to(device)
        target_ids = batch['target_ids'].to(device)

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = getattr(self, 'use_amp', False) and (self.scaler is not None)
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(
                    input_ids=input_ids,
                    input_spectra=input_spectra,
                    special_token_positions=special_token_positions,
                )
                loss, acc = self.get_loss(outputs['logits'], target_ids)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                input_ids=input_ids,
                input_spectra=input_spectra,
                special_token_positions=special_token_positions,
            )
            loss, acc = self.get_loss(outputs['logits'], target_ids)
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss, acc, outputs['h']

    def eval_batch(self, batch: Dict[str, Any], batch_idx: int, device: torch.device):
        input_ids = batch['input_ids'].to(device)
        input_spectra = batch['masked_spectra'].to(device)
        special_token_positions = batch['feature_start_indices'].to(device)
        target_ids = batch['target_ids'].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=getattr(self, 'use_amp', False)):
                outputs = self.model(
                    input_ids=input_ids,
                    input_spectra=input_spectra,
                    special_token_positions=special_token_positions,
                )
                loss, acc = self.get_loss(outputs['logits'], target_ids)
        return loss, acc, outputs['h']

