import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        results = []

        with torch.no_grad():
            current = x

            for layer in model:
                current = layer(current)

                if isinstance(layer, nn.Linear):
                    activations = current

                    mean = activations.mean().item()
                    std = activations.std().item()

                    dead_neurons = (activations <= 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean().item()

                    results.append({"mean": round(mean, 4), "std": round(std, 4), "dead_fraction": round(dead_fraction, 4)})

        return results
                

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        forward = model(x)
        criterion = nn.MSELoss()
        loss = criterion(forward, y)
        loss.backward()
        results = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                results.append({"mean": round(grad.mean().item(), 4), "std": round(grad.std().item(), 4), "norm": round(grad.norm().item(), 4)})

        return results

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        n = len(gradient_stats)

        for i in range(n):
            if activation_stats[i]['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for i in range(n):
            if gradient_stats[i]['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats[-1]['norm'] < 10**(-5):
            return 'vanishing_gradients'
        
        for i in range(n):
            if activation_stats[i]['std'] < 0.1:
                return 'vanishing_gradients'
            
            if activation_stats[i]['std'] > 10.0:
                return 'exploding_gradients'
        
        return 'healthy'