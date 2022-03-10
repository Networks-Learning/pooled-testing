# Pooled Testing of Traced Contacts Under Superspreading Dynamics

This repository contains the code used in the paper *Pooled Testing of Traced Contacts Under Superspreading Dynamics* (provisionally accepted at PLOS Computational Biology). A preliminary version of it is available on [arXiv](https://arxiv.org/abs/2106.15988).

## Dependencies

All the experiments were performed using Python 3.9. In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Code organization

The directory [src](src/) contains the source code for performing simulation experiments with our method and classic Dorfman's method.

The directory [scripts](scripts/) contains bash scripts that use the aforementioned code and pass parameter values required for the various experiments.

The directory [notebooks](notebooks/) contains jupyter notebooks producing the figures appearing in the paper. Each notebook reads output files which first need to be generated by executing the corresponding script of the same name.

The directory [figures](figures/) is used for saving the figures produced by the notebooks.

The directory [outputs](outputs/) is used for saving the json outputs produced by the scripts.

The directory [temp-outputs](temp-outputs/) is used for saving intermediate output files used by the notebooks.

## Citation

If you use parts of the code in this repository for your own research, please consider citing:

    @article{tsirtsis2021group,
        title={Group Testing under Superspreading Dynamics},
        author={Tsirtsis, Stratis and De, Abir and Lorch, Lars and Gomez-Rodriguez, Manuel},
        journal={arXiv preprint arXiv:2106.15988},
        year={2021}
    }
