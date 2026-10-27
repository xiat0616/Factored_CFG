import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from pyro.nn import DenseNN
from pyro.infer.reparam.transform import TransformReparam
from pyro.distributions.conditional import ConditionalTransformedDistribution

from pgm_layers import (
    CNN,
    ConditionalGumbelMax,
    ConditionalTransformedDistributionGumbelMax,
)


class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)  # model defined in parent class
        return samples

    def infer_exogeneous(self, obs):
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()
        # assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)

            if hasattr(self, "discrete_variables"):  # hack for MIMIC
                if (
                    "age" not in intervention.keys()
                    and "finding" not in intervention.keys()
                ):
                    counterfactuals["finding"] = obs["finding"]
                    
            if hasattr(self, "discrete_variables"):  # hack for EMBED
                if (
                    "age" not in intervention.keys()
                    and "density" not in intervention.keys()
                ):
                    counterfactuals["density"] = obs["density"]

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class CelebaFlowPGM(BasePGM):
    def __init__(self, args):
        super().__init__()

        self.variables = {
            "Smiling": "binary",
            "Eyeglasses": "binary",
        }

        # Optional: keep this if you want BasePGM to know these are discrete
        self.discrete_variables = {
            "Smiling": "binary",
            "Eyeglasses": "binary",
        }

        # Learnable priors
        self.smiling_logit = nn.Parameter(torch.tensor([0.0]))
        self.eyeglasses_logit = nn.Parameter(torch.tensor([0.0]))

        if args.pgm_setup != "sup_pgm":
            from pgm_resnet import ResNet, ResNet18, CustomBlock

            shared_model = ResNet(
                CustomBlock,
                layers=[2, 2, 2, 2],
                widths=[64, 128, 256, 512],
                norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),
            )

            shared_model.conv1 = nn.Conv2d(
                args.in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

            kwargs = {
                "in_shape": (args.in_channels, *(args.image_size,) * 2),
                "base_model": shared_model,
            }

            self.encoder_smiling = ResNet18(num_outputs=1, **kwargs)
            self.encoder_eyeglasses = ResNet18(num_outputs=1, **kwargs)

            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )

        else:
            input_shape = (args.in_channels, args.image_size, args.image_size)

            self.encoder_smiling = CNN(input_shape, num_outputs=1)
            self.encoder_eyeglasses = CNN(input_shape, num_outputs=1)

    def model(self, t=None):
        pyro.module("CelebaFlowPGM", self)

        smiling = pyro.sample(
            "Smiling",
            dist.Bernoulli(logits=self.smiling_logit).to_event(1)
        )

        eyeglasses = pyro.sample(
            "Eyeglasses",
            dist.Bernoulli(logits=self.eyeglasses_logit).to_event(1)
        )

        return {
            "Smiling": smiling,
            "Eyeglasses": eyeglasses,
        }

    def guide(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            if obs.get("Smiling") is None:
                smiling_logit = self.encoder_smiling(obs["x"])
                obs["Smiling"] = pyro.sample(
                    "Smiling",
                    dist.Bernoulli(logits=smiling_logit).to_event(1)
                )

            if obs.get("Eyeglasses") is None:
                eyeglasses_logit = self.encoder_eyeglasses(obs["x"])
                pyro.sample(
                    "Eyeglasses",
                    dist.Bernoulli(logits=eyeglasses_logit).to_event(1)
                )

    def model_anticausal(self, **obs):
        pyro.module("CelebaFlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            smiling_logit = self.encoder_smiling(obs["x"])
            q_smiling = dist.Bernoulli(logits=smiling_logit).to_event(1)
            pyro.sample("Smiling_aux", q_smiling, obs=obs["Smiling"])

            eyeglasses_logit = self.encoder_eyeglasses(obs["x"])
            q_eyeglasses = dist.Bernoulli(logits=eyeglasses_logit).to_event(1)
            pyro.sample("Eyeglasses_aux", q_eyeglasses, obs=obs["Eyeglasses"])

    def predict(self, **obs):
        smiling_logit = self.encoder_smiling(obs["x"])
        smiling_prob = torch.sigmoid(smiling_logit)

        eyeglasses_logit = self.encoder_eyeglasses(obs["x"])
        eyeglasses_prob = torch.sigmoid(eyeglasses_logit)

        return {
            "Smiling": smiling_prob,
            "Eyeglasses": eyeglasses_prob,
        }

    def predict_unnorm(self, **obs):
        smiling_logit = self.encoder_smiling(obs["x"])
        eyeglasses_logit = self.encoder_eyeglasses(obs["x"])

        return {
            "Smiling": smiling_logit,
            "Eyeglasses": eyeglasses_logit,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass