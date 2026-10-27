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
    CNN,  # fmt: skip
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

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class EmbedFlowPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "density": "binary",
            "circle": "binary",
        }

        self.density_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))
        self.circle_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))

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

            self.encoder_d = ResNet18(num_outputs=1, **kwargs)
            self.encoder_c = ResNet18(num_outputs=1, **kwargs)

        else:
            if args.enc_net == "cnn":
                input_shape = (1, args.image_size, args.image_size)
                self.encoder_d = CNN(input_shape, num_outputs=1)
                self.encoder_c = CNN(input_shape, num_outputs=1)

    def model(self, t=None):
        pyro.module("EmbedPGM", self)

        pdensity = dist.Bernoulli(logits=self.density_logit).to_event(1)
        density = pyro.sample("density", pdensity)

        pcircle = dist.Bernoulli(logits=self.circle_logit).to_event(1)
        circle = pyro.sample("circle", pcircle)

        return {
            "density": density,
            "circle": circle,
        }


    def guide(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            if obs["density"] is None:
                d_prob = torch.sigmoid(self.encoder_d(obs["x"]))
                qd_x = dist.Bernoulli(probs=d_prob).to_event(1)
                obs["density"] = pyro.sample("density", qd_x)

            if obs["circle"] is None:
                c_prob = torch.sigmoid(self.encoder_c(obs["x"]))
                qc_x = dist.Bernoulli(probs=c_prob).to_event(1)
                pyro.sample("circle", qc_x)


    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classifiers
        pyro.module("EmbedPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(density | x)
            d_prob = torch.sigmoid(self.encoder_d(obs["x"]))
            qd_x = dist.Bernoulli(probs=d_prob).to_event(1)
            pyro.sample("density_aux", qd_x, obs=obs["density"])

            # q(circle | x)
            c_prob = torch.sigmoid(self.encoder_c(obs["x"]))
            qc_x = dist.Bernoulli(probs=c_prob).to_event(1)
            pyro.sample("circle_aux", qc_x, obs=obs["circle"])


    def predict(self, **obs):
        return {
            "density": torch.sigmoid(self.encoder_d(obs["x"])),
            "circle": torch.sigmoid(self.encoder_c(obs["x"])),
        }

    def predict_unnorm(self, **obs):
        return {
            "density": self.encoder_d(obs["x"]),
            "circle": self.encoder_c(obs["x"]),
        }

        
    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass