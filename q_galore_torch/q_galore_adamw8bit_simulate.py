from bitsandbytes.optim.optimizer import Optimizer2State
import bitsandbytes.functional as F

import pdb 
import torch
import torch.distributed as dist

from .q_galore_projector_simulate import GaLoreProjector


class AdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged )

    @torch.no_grad()
    def step(self, closure=None, exchange_step=0):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        #if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"],
                                                quant=group["quant"], group_size=group["quant_group_size"], n_bit=group["quant_n_bit"], 
                                                cos_threshold=group["cos_threshold"], gamma_proj=group["gamma_proj"], queue_size=group["queue_size"])

                    if 'weight_decay' in group and group['weight_decay'] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group['weight_decay_saved'] = group['weight_decay']
                        group['weight_decay'] = 0

                    # low-rank gradient projection
                    grad = state["projector"].project(p.grad, state["step"])

                    # suboptimal implementation
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()

                    p.grad = grad

                if 'state1' not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()

                # GaLore Projection Back
                if "rank" in group:
                    # p.data.zero_() + self.update_step -> p.data = weight_update (torch.float)
                    p.data = p.saved_data.add_(state["projector"].project_back(p.data))  
                    del p.saved_data

                    # apply weight decay
                    if 'weight_decay_saved' in group:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay_saved'])
                        group['weight_decay'] = group['weight_decay_saved']
                        del group['weight_decay_saved']

                if hasattr(p, "group_size"):
                    # quantize back to int8 (simulation)
                    saved_data = p.data.clone()
                    if p.stochastic_round:
                        p.data = self._quantize_stochastic_round(saved_data, q_group_size=p.group_size)
                    else:
                        p.data = self._quantize(saved_data, q_group_size=p.group_size)

        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss

    @torch.no_grad()
    def _quantize(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)

        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
        w = (w - zeros) * scales
        w = w.reshape(org_w_shape)

        return w


    @torch.no_grad()
    def _quantize_stochastic_round(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # Stochastic Rounding
        w_round = w / scales
        up_round_w = torch.ceil(w_round)
        down_round_w = torch.floor(w_round)
        probability = (w_round - down_round_w)
        random = torch.rand_like(probability)
        w = torch.where(random < probability, up_round_w, down_round_w)

        # # Random Rounding
        # w_round = w / scales
        # up_round_w = torch.ceil(w_round)
        # down_round_w = torch.floor(w_round)
        # random = torch.rand_like(up_round_w)
        # w = torch.where(random < 0.5, up_round_w, down_round_w)

        w = torch.clamp(w + zeros, min_int, max_int)
        w = (w - zeros) * scales
        w = w.reshape(org_w_shape)

        return w

