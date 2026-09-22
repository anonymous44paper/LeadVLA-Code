"""Shared-query control with selective semantic coupling (Appendix A.2)."""

import torch
from torch import nn

from .path_composer import ArcLengthPathComposer
from .targets import CLASSES as SEMANTIC_CLASSES


class FollowerHeads(nn.Module):
    """All six objectives supervise pooled queries; only two modulate Residual."""

    def __init__(self, hidden_size, auxiliary_size=256):
        super().__init__()
        if hidden_size <= 0 or auxiliary_size <= 0:
            raise ValueError("Feature dimensions must be positive")
        self.hidden_size = hidden_size
        self.refer_encoder = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, auxiliary_size), nn.SiLU())
        self.formation_encoder = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, auxiliary_size), nn.SiLU())
        self.bbox_head = nn.Linear(auxiliary_size, 4)
        self.lateral_head = nn.Linear(auxiliary_size, 3)
        self.distance_head = nn.Linear(auxiliary_size, 3)
        self.observation_head = nn.Linear(auxiliary_size, 4)
        self.motion_head = nn.Linear(auxiliary_size, 3)
        self.formation_head = nn.Linear(auxiliary_size * 2, 3)
        self.residual_fusion = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, auxiliary_size), nn.SiLU(), nn.Linear(auxiliary_size, hidden_size),
        )
        nn.init.zeros_(self.residual_fusion[-1].weight)
        nn.init.zeros_(self.residual_fusion[-1].bias)

    def modulation(self, predictions):
        soft = torch.cat((predictions["lateral"].softmax(-1), predictions["formation"].softmax(-1)), dim=-1)
        return self.residual_fusion(soft)

    def forward(self, queries):
        if queries.ndim != 3 or queries.shape[1:] != (24, self.hidden_size):
            raise ValueError("Expected action queries [B,24,hidden_size]")
        pooled = queries.mean(1).to(dtype=self.bbox_head.weight.dtype)
        refer = self.refer_encoder(pooled)
        formation = self.formation_encoder(pooled)
        result = {
            "bbox": self.bbox_head(refer).sigmoid(),
            "lateral": self.lateral_head(refer),
            "distance": self.distance_head(refer),
            "observation": self.observation_head(refer),
            "motion": self.motion_head(refer),
            "formation": self.formation_head(torch.cat((refer, formation), dim=-1)),
        }
        delta = self.modulation(result)
        result.update(pace_queries=queries, residual_queries=queries + delta.to(queries.dtype).unsqueeze(1), modulation=delta)
        return result


class ControlHeads(nn.Module):
    """Pace + semantic heads with an explicitly supplied Residual decoder.

    residual_decoder(queries) must return [B,24,3]. Its architecture and weights
    are supplied by the application, not silently replaced by a dummy model.
    """

    def __init__(self, hidden_size, residual_decoder, auxiliary_size=256, pace_hidden_size=None):
        super().__init__()
        self.semantics = FollowerHeads(hidden_size, auxiliary_size)
        self.residual_decoder = residual_decoder
        size = pace_hidden_size or max(256, hidden_size // 4)
        self.pace_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, size), nn.SiLU(), nn.Linear(size, 1))
        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.zeros_(self.pace_head[-1].bias)

    def forward(self, queries):
        output = self.semantics(queries)
        dtype = self.pace_head[-1].weight.dtype
        output["pace"] = self.pace_head(queries.mean(1).to(dtype)).sigmoid().squeeze(-1)
        parameters = list(self.residual_decoder.parameters())
        residual_dtype = parameters[0].dtype if parameters else queries.dtype
        output["residual"] = self.residual_decoder(output["residual_queries"].to(residual_dtype))
        if output["residual"].shape != (queries.shape[0], 24, 3):
            raise ValueError("Residual decoder must return [B,24,3]")
        return output


def gather_action_queries(hidden, input_ids, action_token_id):
    """Gather exactly 24 action-token states from each sequence in one batch."""
    if hidden.ndim != 3 or input_ids.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [B,L,D] and IDs [B,L]")
    mask = input_ids == action_token_id
    if not torch.all(mask.sum(1) == 24):
        raise ValueError("Each sequence must contain exactly 24 action query tokens")
    return hidden[mask].reshape(hidden.shape[0], 24, hidden.shape[-1])


class LeadVLA(nn.Module):
    """One multimodal forward pass followed by shared-query control heads.

    The supplied backbone returns .hidden_states and accepts tokenized multimodal
    inputs. Tokenizer integration, backbone and Residual decoder are external.
    """

    def __init__(self, backbone, residual_decoder, hidden_size, action_token_id, auxiliary_size=256):
        super().__init__()
        self.backbone = backbone
        self.action_token_id = action_token_id
        self.heads = ControlHeads(hidden_size, residual_decoder, auxiliary_size)
        self.composer = ArcLengthPathComposer()

    def forward(self, inputs, route, progress, origin, executed_pace, reaction_valid=None):
        reserved = {"output_hidden_states", "output_attentions", "return_dict"}
        if reserved.intersection(inputs):
            raise ValueError("Backbone output flags are controlled by LeadVLA")
        encoded = self.backbone(**inputs, output_hidden_states=True, output_attentions=False, return_dict=True)
        queries = gather_action_queries(encoded.hidden_states[-1], inputs["input_ids"], self.action_token_id)
        output = self.heads(queries)
        pace = output["pace"].float()
        if reaction_valid is not None:
            if reaction_valid.shape != pace.shape:
                raise ValueError("reaction_valid must be [B]")
            pace = torch.where(reaction_valid.bool(), pace, torch.ones_like(pace))
        composed = self.composer(route.float(), progress.float(), pace, executed_pace.float(), output["residual"].float(), path_origin=origin.float())
        output.update(composed)
        return output
