import torch
from mmcv.runner import OptimizerHook
from torch.nn.utils import clip_grad


class SeOptimizerHook(OptimizerHook):
    """
        175
        neck.fpnse_convs.0.w
        torch.Size([768, 256, 5, 5])
        176
        neck.fpnse_convs.1.w
        torch.Size([768, 256, 5, 5])
        177
        neck.fpnse_convs.2.w
        torch.Size([768, 256, 5, 5])
        178
        neck.fpnse_convs.3.w
        torch.Size([768, 256, 5, 5])
    """
    def clip_grads(self, params):
        filtered_params=[]
        for i, p in enumerate(params):
            if p.requires_grad and p.grad is not None:
                assert not torch.any(torch.isnan(p.grad)), str(i)
                if i in [175, 176, 177, 178]:
                    p.grad[256:-256, ] = p.grad[256:-256, ] / 10.  # w2
                    p.grad[-256:, ] = p.grad[-256:, ] / 20.  # w3
                filtered_params.append(p)
        params = filtered_params
        # params = list(
        #     filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)
