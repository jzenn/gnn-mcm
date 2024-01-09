from typing import Dict, Tuple, Union

import torch

from src.factor.gaussian.variational_gaussian import VariationalGaussianFactor
from src.loss.pmf.pmf_loss import PMFLoss


class GaussianPMFVIEntropyLoss(PMFLoss):
    def compute_elbo(
        self,
        x: torch.Tensor,
        solutes_factor: VariationalGaussianFactor,
        solvents_factor: VariationalGaussianFactor,
        result: torch.Tensor,
        weights: Dict[str, Union[float, torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # entropy
        per_sample_solutes_entropy = solutes_factor.compute_entropy(reduction="none")
        solutes_entropy = (weights["solutes_weight"] * per_sample_solutes_entropy).sum()

        per_sample_solvents_entropy = solvents_factor.compute_entropy(reduction="none")
        solvents_entropy = (
            weights["solvents_weight"] * per_sample_solvents_entropy
        ).sum()

        # samples
        solutes_sample = solutes_factor.get_last_sample()
        solvents_sample = solvents_factor.get_last_sample()

        # data log-likelihood
        per_sample_data_log_likelihood = self.data_log_likelihood(
            x.repeat(1, self.number_of_samples_for_expectation),
            result,
            self.get_data_likelihood_std(),
        )
        data_log_likelihood = (
            weights["data_log_likelihood_weight"]
            * per_sample_data_log_likelihood.mean(-1).sum()
        )

        # prior log-likelihoods
        per_sample_solutes_log_prior = self.log_prior(
            solutes_sample,
            self.solutes_prior_factor.get_mean().unsqueeze(-2),
            self.solutes_prior_factor.get_std(get_as="tril").unsqueeze(-3),
        )
        solutes_log_prior = (
            weights["solutes_weight"] * per_sample_solutes_log_prior.mean(-1)
        ).sum()

        per_sample_solvents_log_prior = self.log_prior(
            solvents_sample,
            self.solvents_prior_factor.get_mean().unsqueeze(-2),
            self.solvents_prior_factor.get_std(get_as="tril").unsqueeze(-3),
        )
        solvents_log_prior = (
            weights["solvents_weight"] * per_sample_solvents_log_prior.mean(-1)
        ).sum()

        # weight for log-likelihood and KL divergence
        kl_weight = weights["kl_weight"]
        likelihood_weight = weights["likelihood_weight"]

        return (
            solutes_entropy,
            solvents_entropy,
            data_log_likelihood,
            solutes_log_prior,
            solvents_log_prior,
            likelihood_weight * data_log_likelihood
            + kl_weight
            * (
                solutes_log_prior
                + solvents_log_prior
                + solutes_entropy
                + solvents_entropy
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        solutes_factor: VariationalGaussianFactor,
        solvents_factor: VariationalGaussianFactor,
        result: torch.Tensor,
        weights: Dict[str, Union[float, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        (
            solutes_entropy,
            solvents_entropy,
            data_log_likelihood,
            solutes_log_prior,
            solvents_log_prior,
            elbo,
        ) = self.compute_elbo(x, solutes_factor, solvents_factor, result, weights)

        result_mean = result.mean(1).unsqueeze(1)
        mse_loss = self.get_mse_loss(result_mean, x)
        mae_loss = self.get_mae_loss(result_mean, x)
        min_abs_loss, max_abs_loss = self.get_min_max_abs_loss(result_mean, x)

        # weight for loss
        loss_weight = weights["loss_weight"]

        return {
            "solutes_entropy": solutes_entropy,
            "solvents_entropy": solvents_entropy,
            "data_log_likelihood": data_log_likelihood,
            "solutes_log_prior": solutes_log_prior,
            "solvents_log_prior": solvents_log_prior,
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
