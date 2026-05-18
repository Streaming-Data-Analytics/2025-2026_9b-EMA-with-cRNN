import torch

# Function for the EMA weight update: apply the EMA formula:
# θ_EMA = λ * θ_EMA + (1 - λ) * θ_online
def update_ema_weights(ema_model, online_model, lam):
    with torch.no_grad():
        for ema_param, online_param in zip(ema_model.parameters(), online_model.parameters()):
            ema_param.data.mul_(lam).add_(online_param.data * (1 - lam))
