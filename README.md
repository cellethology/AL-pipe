# Installation
----
## Installing from source
```
conda create --name AL-pipeline python=3.13
conda activate AL-pipeline
pip install uv
git clone git@github.com:cellethology/AL-pipe.git
cd AL-pipe
uv sync --dev
```

# TODO 
------
- [ ] Double check the classes in python is in adherence to the public and private syntax
- [ ] Beware of the `# @package _global_` namespace in hydra yaml config
- [ ] Write plotting function integrated with the trainer
- [ ] Check all the todos
- [ ] Update the codespace with datamodule class (easier to manage data) `trainer.test(model, datamodule=dm)` 
- [ ] Double check if the seed is correct