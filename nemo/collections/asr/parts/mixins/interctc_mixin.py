# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Tuple

import torch

from nemo.core.classes.mixins import AccessMixin


class InterCTCMixin:
    """Adds utilities for computing interCTC loss from https://arxiv.org/abs/2102.03216.

    To use, make sure encoder defines ``capture_output_at_layers (list[int])``
    property and registers layer_output_X and layer_length_X for all layers that
    we want to get loss from. Then call

        * ``setup_interctc`` in the init method
        * ``self.add_interctc_losses`` after computing regular loss.
        * ``self.finalize_interctc_metrics(metrics, outputs, prefix="val_")``
          in the `multi_validation_epoch_end` method.
        * ``self.finalize_interctc_metrics(metrics, outputs, prefix="test_")``
          in the `multi_test_epoch_end` method.
    """

    def setup_interctc(self):
        self.intermediate_loss_weights = self.cfg.get('intermediate_loss_weights', [])
        self.main_loss_weight = 1.0 - sum(self.intermediate_loss_weights)
        if self.main_loss_weight <= 0.0:
            raise ValueError(
                "Make sure that sum of intermediate loss weights is < 1.0. "
                "Note that we don't do any normalization and assign "
                "remaining weight to the regular model loss. "
                "E.g., if intermediate_loss_weights = [0.1, 0.3], regular "
                "loss will have weight of 0.6"
            )
        self.interctc_enabled = len(self.intermediate_loss_weights) > 0
        if self.interctc_enabled and not hasattr(self.encoder, 'capture_output_at_layers'):
            raise ValueError('To use intermediate CTC loss, encoder has to define "capture_output_at_layers" property')

        if len(self.encoder.capture_output_at_layers) != len(self.intermediate_loss_weights):
            raise ValueError('Length of encoder.capture_output_at_layers has to match intermediate_loss_weights')

    def finalize_interctc_metrics(self, metrics, outputs, prefix):
        if self.interctc_enabled:
            for layer_idx in self.encoder.capture_output_at_layers:
                loss = torch.stack([x[f"{prefix}inter_ctc_loss_l{layer_idx}"] for x in outputs]).mean()
                wer_num = torch.stack([x[f"{prefix}inter_wer_num_l{layer_idx}"] for x in outputs]).sum()
                wer_denom = torch.stack([x[f"{prefix}inter_wer_denom_l{layer_idx}"] for x in outputs]).sum()
                metrics["log"].update(
                    {
                        f"{prefix}inter_ctc_loss_l{layer_idx}": loss,
                        f"{prefix}inter_wer_l{layer_idx}": wer_num / wer_denom,
                    }
                )
            metrics["log"][f"{prefix}final_ctc_loss"] = torch.stack(
                [x[f"{prefix}final_ctc_loss"] for x in outputs]
            ).mean()

    def get_captured_tensors(self):
        if not self.interctc_enabled:
            return []
        # if intermediate_loss_weights was set, the encoder has to register
        # layer_output_X and layer_length_X tensors. We need to apply decoder
        # to each of them and compute CTC loss.
        module_registry = AccessMixin.get_module_registry(self.encoder)['']  # key for encoder
        captured_tensors = []
        for layer_idx in self.encoder.capture_output_at_layers:
            try:
                layer_outputs = module_registry[f"layer_output_{layer_idx}"]
                layer_lengths = module_registry[f"layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! "
                    "Check if length of model.encoder.captured_layer_outputs matches "
                    "length of model.intermediate_loss_weights properties."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError(
                    "Make sure encoder.forward is called exactly one time before interCTC loss is computed."
                )
            captured_tensors.append((self.decoder(encoder_output=layer_outputs[0]), layer_lengths[0]))
        return captured_tensors

    def add_interctc_losses(
        self,
        loss_value: torch.tensor,
        transcript: torch.tensor,
        transcript_len: torch.tensor,
        compute_wer: bool,
        log_wer_num_denom: bool = False,
        log_prefix: str = "",
    ) -> Tuple[torch.Tensor, Dict]:
        """Adding interCTC losses if required."""
        if not self.interctc_enabled or not AccessMixin.is_access_enabled():
            return loss_value, {}
        metrics = {f"{log_prefix}final_ctc_loss": loss_value}
        captured_tensors = self.get_captured_tensors()

        for layer_idx, intermediate_result, loss_weight in zip(
            self.encoder.capture_output_at_layers, captured_tensors, self.intermediate_loss_weights
        ):
            inter_loss_value = self.loss(
                log_probs=intermediate_result[0],
                targets=transcript,
                target_lengths=transcript_len,
                input_lengths=intermediate_result[1],
            )
            metrics[f"{log_prefix}inter_ctc_loss_l{layer_idx}"] = loss_value.detach()
            loss_value += inter_loss_value * loss_weight
            if compute_wer:
                self._wer.update(
                    predictions=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=intermediate_result[1],
                )
                wer, wer_num, wer_denom = self._wer.compute()
                self._wer.reset()
                metrics.update({f'{log_prefix}inter_wer_l{layer_idx}': wer})
                if log_wer_num_denom:
                    metrics.update(
                        {
                            f'{log_prefix}inter_wer_num_l{layer_idx}': wer_num,
                            f'{log_prefix}inter_wer_denom_l{layer_idx}': wer_denom,
                        }
                    )

        # return total loss
        return loss_value, metrics
