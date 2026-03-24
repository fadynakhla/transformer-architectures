import threading

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from mlflow import MlflowClient


def log_grad_norm(model_grads: dict[str, torch.Tensor], step: int) -> None:
    total_grad_norm = sum(float(torch.norm(g)) ** 2 for g in model_grads.values())
    mlflow.log_metric("grad_norm", total_grad_norm**0.5, step=step)


def log_param_and_grad_figures(
    client: MlflowClient,
    run_id: str,
    model_params: dict[str, torch.Tensor],
    model_grads: dict[str, torch.Tensor],
    step: int,
) -> None:
    for name, param in model_params.items():
        path = name.replace(".", "/")

        fig, ax = plt.subplots()
        ax.hist(param.flatten().numpy(), bins=64)
        ax.set_title(name)
        client.log_figure(run_id, fig, f"params/step_{step}/{path}.png")
        plt.close(fig)

        if (grad_name := f"grad/{name}") in model_grads:
            grads = model_grads[grad_name]

            fig, ax = plt.subplots()
            ax.hist(grads.flatten().numpy(), bins=64)
            ax.set_title(f"grad/{name}")
            client.log_figure(run_id, fig, f"grads/step_{step}/{path}.png")
            plt.close(fig)


def log_grads(model: nn.Module, step: int, log_distributions: bool = False) -> None:
    run = mlflow.active_run()
    if run is None:
        raise ValueError("There is no active mlflow run.")
    run_id = run.info.run_id
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

    log_grad_norm(model_grads, step)

    if log_distributions:
        threading.Thread(
            target=log_param_and_grad_figures,
            args=(client, run_id, model_params, model_grads, step),
        ).start()
