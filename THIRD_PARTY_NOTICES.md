# Third-party notices

The project license covers original contributions. The following upstream notices also apply to adapted code; copies of their license texts are included in `licenses/`.

| Upstream | Relevant code | License notice |
| --- | --- | --- |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | Diffusion infrastructure, U-Net, timestep respacing, losses, and neural-network utilities | [MIT, OpenAI](licenses/guided-diffusion.txt) |
| [causal-gen](https://github.com/biomedia-mira/causal-gen) | Pyro graphical-model and predictor utilities in `causal_model/pgm/` | [MIT, Fabio De Sousa Ribeiro](licenses/causal-gen.txt) |
| [ema-pytorch](https://github.com/lucidrains/ema-pytorch) | EMA implementation in `guided_diffusion/nn_utils.py` | [MIT, Phil Wang](licenses/ema-pytorch.txt) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | Attention pooling in `guided_diffusion/nn_utils.py` | [MIT, OpenAI](licenses/CLIP.txt) |
| [torchvision](https://github.com/pytorch/vision) | ResNet implementation in `causal_model/pgm/pgm_resnet.py` | [BSD 3-Clause](licenses/torchvision.txt) |

The diffusion source also retains its historical references to [Ho et al.'s original diffusion implementation](https://github.com/hojonathanho/diffusion). Installed dependencies, datasets, pretrained evaluation networks, and any separately obtained checkpoints retain their respective licenses and terms.
