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
            "Mouth_Slightly_Open": "binary",
            "Male": "binary",
            "Young": "binary",
        }

        # This helps the BasePGM recognize discrete variables for special treatment
        self.discrete_variables = {"Mouth_Slightly_Open": "binary"}
        
        # Learnable prior for Smiling
        self.smiling_logit = nn.Parameter(torch.tensor([0.0]))  # p(smile=1)

        # Conditional logits for Mouth_Slightly_Open | Smiling
        self.mouth_logit_0 = nn.Parameter(torch.tensor([-1.0]))  # p(mouth_open=1 | smile=0)
        self.mouth_logit_1 = nn.Parameter(torch.tensor([2.0]))   # p(mouth_open=1 | smile=1)

        # Learnable prior for Male, Young
        self.male_logit = nn.Parameter(torch.tensor([0.0]))  
        self.young_logit = nn.Parameter(torch.tensor([0.0]))
        
        if args.pgm_setup != "sup_pgm":
            from pgm_resnet import ResNet, ResNet18, CustomBlock

            # Shared ResNet backbone
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

            # q(Smiling | x) ~ Bernoulli
            self.encoder_smiling = ResNet18(num_outputs=1, **kwargs)

            # q(Mouth_Slightly_Open | x, Smiling) ~ Bernoulli
            self.encoder_mouth = ResNet18(num_outputs=1, context_dim=1, **kwargs)

            # Independent root nodes
            self.encoder_male = ResNet18(num_outputs=1, **kwargs)
            self.encoder_young = ResNet18(num_outputs=1, **kwargs)

            # Softplus scale helper (if needed for normal vars)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )

        else:
            # Simpler CNN-based encoders
            input_shape = (args.in_channels, args.image_size, args.image_size)

            self.encoder_smiling = CNN(input_shape, num_outputs=1)
            self.encoder_mouth = CNN(input_shape, num_outputs=1, context_dim=1)

            self.encoder_male = CNN(input_shape, num_outputs=1)
            self.encoder_young = CNN(input_shape, num_outputs=1)

    def model(self, t=None):
        pyro.module("CelebaFlowPGM", self)

        # p(Smiling)
        smiling = pyro.sample("Smiling", dist.Bernoulli(logits=self.smiling_logit).to_event(1))

        # p(Mouth_Slightly_Open | Smiling)
        logit_mouth = smiling * self.mouth_logit_1 + (1 - smiling) * self.mouth_logit_0
        pyro.sample("Mouth_Slightly_Open", dist.Bernoulli(logits=logit_mouth).to_event(1))

        # Independent binary variables
        pyro.sample("Male", dist.Bernoulli(logits=self.male_logit).to_event(1))
        pyro.sample("Young", dist.Bernoulli(logits=self.young_logit).to_event(1))

        return {
            "Smiling": smiling,
            "Mouth_Slightly_Open": logit_mouth.sigmoid(),  # optional
        }

    def guide(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            if obs.get("Smiling") is None:
                logit_smile = self.encoder_smiling(obs["x"])
                obs["Smiling"] = pyro.sample("Smiling", dist.Bernoulli(logits=logit_smile).to_event(1))

            if obs.get("Mouth_Slightly_Open") is None:
                logit_mouth = self.encoder_mouth(obs["x"], y=obs["Smiling"])
                pyro.sample("Mouth_Slightly_Open", dist.Bernoulli(logits=logit_mouth).to_event(1))

            for var, encoder in zip(
                ["Male", "Young"],
                [self.encoder_male, self.encoder_young]
            ):
                if obs.get(var) is None:
                    logits = encoder(obs["x"])
                    pyro.sample(var, dist.Bernoulli(logits=logits).to_event(1))
    def model_anticausal(self, **obs):
        pyro.module("CelebaFlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(Smiling | x) ~ Bernoulli
            smile_logit = self.encoder_smiling(obs["x"])
            q_smile = dist.Bernoulli(logits=smile_logit).to_event(1)
            pyro.sample("Smiling_aux", q_smile, obs=obs["Smiling"])

            # q(Mouth_Slightly_Open | x, Smiling)
            mouth_logit = self.encoder_mouth(obs["x"], y=obs["Smiling"])
            q_mouth = dist.Bernoulli(logits=mouth_logit).to_event(1)
            pyro.sample("Mouth_Slightly_Open_aux", q_mouth, obs=obs["Mouth_Slightly_Open"])

            # q(Male | x) ~ Bernoulli
            male_logit = self.encoder_male(obs["x"])
            q_male = dist.Bernoulli(logits=male_logit).to_event(1)
            pyro.sample("Male_aux", q_male, obs=obs["Male"])

            # q(Young | x) ~ Bernoulli
            young_logit = self.encoder_young(obs["x"])
            q_young = dist.Bernoulli(logits=young_logit).to_event(1)
            pyro.sample("Young_aux", q_young, obs=obs["Young"])

    def predict(self, **obs):
        smiling_logit = self.encoder_smiling(obs["x"])
        smiling_prob = torch.sigmoid(smiling_logit)

        mouth_logit = self.encoder_mouth(obs["x"], y=smiling_prob)
        mouth_prob = torch.sigmoid(mouth_logit)

        male_logit = self.encoder_male(obs["x"])
        male_prob = torch.sigmoid(male_logit)

        young_logit = self.encoder_young(obs["x"])
        young_prob = torch.sigmoid(young_logit)

        return {
            "Smiling": smiling_prob,
            "Mouth_Slightly_Open": mouth_prob,
            "Male": male_prob,
            "Young": young_prob,
        }

    def predict_unnorm(self, **obs):
        smiling_logit = self.encoder_smiling(obs["x"])
        mouth_logit = self.encoder_mouth(obs["x"], y=obs['Smiling'])  # or sampled
        male_logit = self.encoder_male(obs["x"])
        young_logit = self.encoder_young(obs["x"])

        return {
            "Smiling": smiling_logit,
            "Mouth_Slightly_Open": mouth_logit,
            "Male": male_logit,
            "Young": young_logit,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass  # Required by BasePGM but unused
