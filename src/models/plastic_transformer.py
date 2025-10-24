import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 0:
        return value
    return value.view(-1)


@dataclass
class PlasticLayerRecord:
    layer: "PlasticLinear"
    state: Dict[str, torch.Tensor]
    pre: torch.Tensor
    post: torch.Tensor
    require_grad: bool
    weight_grad: Optional[torch.Tensor] = None
    bias_grad: Optional[torch.Tensor] = None


class PlasticLinear(nn.Module):
    """
    Linear layer with a per-trial plastic component that is updated according to
    either a Hebbian or gradient-based plasticity rule.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha_init: float = 0.02,
        beta_init: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
        self.alpha = nn.Parameter(torch.full((out_features, in_features), alpha_init))
        if bias:
            self.beta = nn.Parameter(torch.full((out_features,), beta_init))
        else:
            self.beta = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def init_state(self, device: torch.device) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {
            "w_plastic": torch.zeros(
                self.out_features, self.in_features, device=device, dtype=self.weight.dtype
            ),
        }
        if self.bias is not None:
            state["b_plastic"] = torch.zeros(
                self.out_features, device=device, dtype=self.weight.dtype
            )
        return state

    def forward_step(
        self,
        x: torch.Tensor,
        state: Dict[str, torch.Tensor],
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, PlasticLayerRecord]:
        """
        Processes a single time step.

        Args:
            x: Input vector of shape (in_features,)
            state: Plastic state dictionary for this layer.
            require_grad: Whether the plastic tensors should require gradients
                for gradient-based plasticity.

        Returns:
            Tuple containing the output vector and the record for later plastic updates.
        """
        if x.dim() != 1:
            raise ValueError("PlasticLinear.forward_step expects 1D input tensor.")

        if require_grad:
            state["w_plastic"] = state["w_plastic"].detach().requires_grad_(True)
            if self.bias is not None:
                state["b_plastic"] = state["b_plastic"].detach().requires_grad_(True)

        total_weight = self.weight + state["w_plastic"]
        bias = None
        if self.bias is not None:
            bias = self.bias + state["b_plastic"]
        y = F.linear(x.unsqueeze(0), total_weight, bias=bias).squeeze(0)

        record = PlasticLayerRecord(
            layer=self,
            state=state,
            pre=x,
            post=y,
            require_grad=require_grad,
        )
        return y, record

    def compute_hebbian_delta(self, record: PlasticLayerRecord) -> torch.Tensor:
        pre = record.pre.detach()
        post = record.post.detach()
        return torch.outer(post, pre)

    def apply_hebbian_update(
        self, record: PlasticLayerRecord, eta: torch.Tensor, delta: torch.Tensor
    ) -> None:
        with torch.no_grad():
            state = record.state
            state["w_plastic"].mul_(1.0 - eta).add_(eta * self.alpha * delta)
            if self.bias is not None:
                delta_b = record.post.detach()
                beta = self.beta
                if beta is None:
                    raise RuntimeError("Bias plasticity requested but beta is None.")
                state["b_plastic"].mul_(1.0 - eta).add_(eta * beta * delta_b)
        record.state["w_plastic"] = record.state["w_plastic"].detach()
        if self.bias is not None:
            record.state["b_plastic"] = record.state["b_plastic"].detach()

    def apply_gradient_update(
        self,
        record: PlasticLayerRecord,
        eta: torch.Tensor,
        weight_grad: Optional[torch.Tensor],
        bias_grad: Optional[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            state = record.state
            if weight_grad is None:
                weight_grad = torch.zeros_like(state["w_plastic"])
            state["w_plastic"].mul_(1.0 - eta).add_(eta * self.alpha * weight_grad)
            if self.bias is not None:
                if bias_grad is None:
                    bias_grad = torch.zeros_like(state["b_plastic"])
                beta = self.beta
                if beta is None:
                    raise RuntimeError("Bias plasticity requested but beta is None.")
                state["b_plastic"].mul_(1.0 - eta).add_(eta * beta * bias_grad)

        if record.require_grad:
            # Detach the tensors so they do not keep references to old graphs.
            record.state["w_plastic"] = record.state["w_plastic"].detach()
            if self.bias is not None:
                record.state["b_plastic"] = record.state["b_plastic"].detach()


class PlasticFeedForward(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = PlasticLinear(dim_model, dim_ff)
        self.fc2 = PlasticLinear(dim_ff, dim_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def init_state(self, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "fc1": self.fc1.init_state(device),
            "fc2": self.fc2.init_state(device),
        }

    def forward_step(
        self,
        x: torch.Tensor,
        state: Dict[str, Dict[str, torch.Tensor]],
        require_grad: bool,
    ) -> Tuple[torch.Tensor, List[PlasticLayerRecord]]:
        records: List[PlasticLayerRecord] = []
        hidden1, rec1 = self.fc1.forward_step(x, state["fc1"], require_grad=require_grad)
        records.append(rec1)
        hidden1 = self.act(hidden1)
        hidden1 = self.dropout(hidden1)
        hidden2, rec2 = self.fc2.forward_step(hidden1, state["fc2"], require_grad=require_grad)
        records.append(rec2)
        hidden2 = self.dropout(hidden2)
        return hidden2, records


class PlasticTransformerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim_model)
        self.feed_forward = PlasticFeedForward(dim_model, dim_ff, dropout)

    def init_state(self, device: torch.device) -> Dict[str, Any]:
        return {
            "history": None,  # (1, steps, dim_model)
            "ff": self.feed_forward.init_state(device),
        }

    def forward_step(
        self,
        x: torch.Tensor,
        state: Dict[str, Any],
        require_grad: bool,
    ) -> Tuple[torch.Tensor, List[PlasticLayerRecord]]:
        residual = x
        x_norm = self.ln1(x)
        q = x_norm.unsqueeze(0).unsqueeze(0)  # (batch=1, seq=1, dim)

        history = state["history"]
        if history is None:
            kv = q
        else:
            kv = torch.cat([history, q], dim=1)

        attn_output, _ = self.self_attn(q, kv, kv, need_weights=False)
        attn_output = attn_output.squeeze(0).squeeze(0)
        x = residual + self.dropout_attn(attn_output)
        state["history"] = kv

        residual = x
        x_norm = self.ln2(x)
        ff_out, records = self.feed_forward.forward_step(
            x_norm, state["ff"], require_grad=require_grad
        )
        x = residual + ff_out
        return x, records


class PlasticTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        aux_dim: int = 4,
        rule: str = "hebbian",
        eta0: float = 0.2,
        max_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if rule not in {"hebbian", "gradient", "none"}:
            raise ValueError(f"Unsupported plasticity rule: {rule}")
        self.rule = rule
        self.aux_dim = aux_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                PlasticTransformerBlock(model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, output_dim + aux_dim + 1)
        self.eta0 = eta0
        self.max_norm = max_norm
        if rule == "gradient":
            internal_dim = output_dim + aux_dim + 1
            self.internal_proj = nn.Linear(internal_dim, internal_dim, bias=False)
        else:
            self.internal_proj = None

    def init_state(self, device: torch.device) -> Dict[str, Any]:
        return {
            "blocks": [block.init_state(device) for block in self.blocks],
            "step": 0,
        }

    def forward_step(
        self,
        x: torch.Tensor,
        state: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single time step (single sample). The caller is responsible for
        resetting plastic states between trials.
        """
        state["step"] += 1
        hidden = self.input_proj(x)

        plastic_records: List[PlasticLayerRecord] = []
        for block, block_state in zip(self.blocks, state["blocks"]):
            hidden, block_records = block.forward_step(
                hidden, block_state, require_grad=self.rule == "gradient"
            )
            plastic_records.extend(block_records)

        hidden = self.final_ln(hidden)
        raw_output = self.head(hidden)
        logits = raw_output[: self.output_dim]
        tilde_eta = raw_output[self.output_dim]
        aux = raw_output[self.output_dim + 1 :]

        if self.rule == "gradient":
            assert self.internal_proj is not None
            concat_vec = torch.cat([logits, aux, tilde_eta.unsqueeze(0)], dim=0)
            internal_vec = self.internal_proj(concat_vec)
            internal_loss = (internal_vec.pow(2).mean())
            grad_targets: List[torch.Tensor] = []
            for record in plastic_records:
                grad_targets.append(record.state["w_plastic"])
                if record.layer.bias is not None:
                    grad_targets.append(record.state["b_plastic"])
            grads = torch.autograd.grad(
                internal_loss,
                grad_targets,
                retain_graph=True,
                allow_unused=True,
            )
            grad_iter = iter(grads)
            delta_norm_sq = torch.tensor(0.0, device=logits.device)
            for record in plastic_records:
                weight_grad = next(grad_iter, None)
                bias_grad = None
                if record.layer.bias is not None:
                    bias_grad = next(grad_iter, None)
                record.weight_grad = (
                    torch.zeros_like(record.state["w_plastic"]) if weight_grad is None else weight_grad
                )
                record.bias_grad = (
                    torch.zeros_like(record.state["b_plastic"])
                    if record.layer.bias is not None and bias_grad is None
                    else bias_grad
                )
                delta_norm_sq = delta_norm_sq + record.weight_grad.pow(2).sum()
            delta_norm = torch.sqrt(delta_norm_sq + 1e-8)
        elif self.rule == "hebbian":
            delta_norm_sq = torch.tensor(0.0, device=logits.device)
            for record in plastic_records:
                delta = record.layer.compute_hebbian_delta(record)
                record.weight_grad = delta  # reuse field for update
                delta_norm_sq = delta_norm_sq + delta.pow(2).sum()
                if record.layer.bias is not None:
                    record.bias_grad = record.post.detach()
            delta_norm = torch.sqrt(delta_norm_sq + 1e-8)
        else:  # rule == "none"
            for record in plastic_records:
                record.state["w_plastic"] = record.state["w_plastic"].detach()
                if record.layer.bias is not None:
                    record.state["b_plastic"] = record.state["b_plastic"].detach()
            eta = torch.tensor(0.0, device=logits.device)
            return {
                "logits": logits,
                "tilde_eta": tilde_eta.detach(),
                "aux": aux,
                "eta": eta,
                "diagnostics": {
                    "plastic_norm": torch.tensor(0.0, device=logits.device),
                },
            }

        scaling = torch.ones_like(tilde_eta)
        if self.max_norm > 0:
            scaling = torch.clamp(self.max_norm / (delta_norm + 1e-8), max=1.0)

        eta = self.eta0 * torch.sigmoid(tilde_eta) * scaling

        for record in plastic_records:
            if self.rule == "gradient":
                record.layer.apply_gradient_update(record, eta, record.weight_grad, record.bias_grad)
            else:
                record.layer.apply_hebbian_update(record, eta, record.weight_grad)

        plastic_norm = torch.tensor(0.0, device=logits.device)
        with torch.no_grad():
            for block_state in state["blocks"]:
                ff_state = block_state["ff"]
                for layer_state in ff_state.values():
                    w_plastic = layer_state.get("w_plastic")
                    if w_plastic is not None:
                        plastic_norm = plastic_norm + w_plastic.pow(2).sum()

        output_dict = {
            "logits": logits,
            "tilde_eta": tilde_eta.detach(),
            "aux": aux,
            "eta": eta.detach(),
            "diagnostics": {
                "plastic_norm": plastic_norm.detach(),
            },
        }
        return output_dict
