import argparse
from typing import Dict, Tuple, Union

import torch

from src.factor.gaussian.variational_gaussian import VariationalGaussianFactor
from src.loss.pmf.pmf_loss import PMFLoss


class GaussianPMFVIKLLoss(PMFLoss):
    def compute_elbo(
        self,
        x: torch.Tensor,
        solutes_factor: VariationalGaussianFactor,
        solvents_factor: VariationalGaussianFactor,
        result: torch.Tensor,
        weights: Dict[str, Union[float, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # kl-divergences
        if self.sample_kl:
            per_sample_solutes_kl = solutes_factor.sample_kl_div(
                self.solutes_prior_factor.log_prob,
                self.number_of_samples_for_expectation
                if self.use_different_likelihood_prior_sample
                else -1,
                reduction="none",
            )
        else:
            per_sample_solutes_kl = solutes_factor.compute_kl_div(
                self.solutes_prior_factor.get_mean(),
                self.solutes_prior_factor.get_std(),
                reduction="none",
            )
        solutes_kl = (weights["solutes_weight"] * per_sample_solutes_kl).sum()

        if self.sample_kl:
            per_sample_solvents_kl = solvents_factor.sample_kl_div(
                self.solvents_prior_factor.log_prob,
                self.number_of_samples_for_expectation
                if self.use_different_likelihood_prior_sample
                else -1,
                reduction="none",
            )
        else:
            per_sample_solvents_kl = solvents_factor.compute_kl_div(
                self.solvents_prior_factor.get_mean(),
                self.solvents_prior_factor.get_std(),
                reduction="none",
            )
        solvents_kl = (weights["solvents_weight"] * per_sample_solvents_kl).sum()

        # data log-likelihood
        per_sample_data_log_likelihood = self.data_log_likelihood(
            x.repeat(1, self.number_of_samples_for_expectation),
            result,
            self.get_data_likelihood_std(),
        )
        data_log_likelihood = (
            weights["data_log_likelihood_weight"]
            * per_sample_data_log_likelihood.mean(1).sum()
        )

        # weight for log-likelihood and KL divergence
        kl_weight = weights["kl_weight"]
        likelihood_weight = weights["likelihood_weight"]

        return (
            solutes_kl,
            solvents_kl,
            data_log_likelihood,
            likelihood_weight * data_log_likelihood
            - kl_weight * (solutes_kl + solvents_kl),
        )

    def forward(
        self,
        x: torch.Tensor,
        solutes_factor: VariationalGaussianFactor,
        solvents_factor: VariationalGaussianFactor,
        result: torch.Tensor,
        weights: Dict[str, Union[float, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        solutes_kl, solvents_kl, data_log_likelihood, elbo = self.compute_elbo(
            x, solutes_factor, solvents_factor, result, weights
        )

        result_mean = result.mean(1).unsqueeze(1)
        mse_loss = self.get_mse_loss(result_mean, x)
        mae_loss = self.get_mae_loss(result_mean, x)
        min_abs_loss, max_abs_loss = self.get_min_max_abs_loss(result_mean, x)

        # weight for loss
        loss_weight = weights["loss_weight"]

        return {
            "solutes_kl": solutes_kl,
            "solvents_kl": solvents_kl,
            "data_log_likelihood": data_log_likelihood,
            "elbo": elbo,
            "mse": mse_loss,
            "mae": mae_loss,
            "ae-min": min_abs_loss,
            "ae-max": max_abs_loss,
            "loss_weight": weights["loss_weight"],
            "kl_weight": weights["kl_weight"],
            "likelihood_weight": weights["likelihood_weight"],
            "loss": loss_weight * (-elbo),
        }
