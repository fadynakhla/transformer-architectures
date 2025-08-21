import threading

import aim
import torch
import torch.nn as nn


def log_gradients_and_params(
    run: aim.Run,
    model_params: dict[str, torch.Tensor],
    model_grads: dict[str, torch.Tensor],
    step: int,
    epoch: int,
) -> None:
    total_grad_norm = 0.0

    for name, param in model_params.items():
        # Log parameter distribution
        params = aim.Distribution(param)
        run.track(params, name=f"params/{name}", step=step, epoch=epoch)

        if (grad_name := f"grad/{name}") in model_grads:
            grads = model_grads[grad_name]
            run.track(
                aim.Distribution(grads), name=f"grads/{name}", step=step, epoch=epoch
            )

            grad_norm = float(torch.norm(grads))
            total_grad_norm += grad_norm**2
            # run.track(grad_norm, name=f"grad_norm/{name}", step=step, epoch=epoch)

    total_grad_norm = total_grad_norm**0.5
    run.track(total_grad_norm, name="grad_norm", step=step, epoch=epoch)


def async_log(run: aim.Run, model: nn.Module, step: int, epoch: int) -> None:
    """Move model to CPU safely and log gradients asynchronously."""
    model_params = {}
    model_grads = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        model_params[name] = param.data.cpu().detach()
        if param.grad is None:
            continue
        model_grads[f"grad/{name}"] = param.grad.cpu().clone().detach()
        # if param.requires_grad:
        #     model_params[name] = param.data.cpu().detach()
        #     if param.grad is not None:
        #         model_grads[f"grad/{name}"] = param.grad.cpu().clone().detach()

    threading.Thread(
        target=log_gradients_and_params,
        args=(run, model_params, model_grads, step, epoch),
    ).start()
