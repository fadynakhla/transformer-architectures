import threading

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from mlflow import MlflowClient


def log_gradients_and_params(
    client: MlflowClient,
    run_id: str,
    model_params: dict[str, torch.Tensor],
    model_grads: dict[str, torch.Tensor],
    step: int,
) -> None:
    total_grad_norm = 0.0

    for name, param in model_params.items():
        fig, ax = plt.subplots()
        ax.hist(param.flatten().numpy(), bins=64)
        ax.set_title(name)
        client.log_figure(run_id, fig, f"params/{name}_step_{step}.png")
        plt.close(fig)

        if (grad_name := f"grad/{name}") in model_grads:
            grads = model_grads[grad_name]

            fig, ax = plt.subplots()
            ax.hist(grads.flatten().numpy(), bins=64)
            ax.set_title(f"grad/{name}")
            client.log_figure(run_id, fig, f"grads/{name}_step_{step}.png")
            plt.close(fig)

            grad_norm = float(torch.norm(grads))
            total_grad_norm += grad_norm**2

    total_grad_norm = total_grad_norm**0.5
    client.log_metric(run_id, "grad_norm", total_grad_norm, step=step)


def async_log(model: nn.Module, step: int) -> None:
    run_id = mlflow.active_run().info.run_id
    client = MlflowClient()

    model_params: dict[str, torch.Tensor] = {}
    model_grads: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        model_params[name] = param.data.cpu().detach()
        if param.grad is None:
            continue
        model_grads[f"grad/{name}"] = param.grad.cpu().clone().detach()

    threading.Thread(
        target=log_gradients_and_params,
        args=(client, run_id, model_params, model_grads, step),
    ).start()
