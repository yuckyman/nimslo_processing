# auto-aligning gif-ifying python code to process nimslo batches

## intuition
- select four to six images from set of developed film
- using convolutional neural networks, align four images to a manually selected point
- match histogram of four images, non-aggressively, to match exposure/brightness
- export gif of four images, aligned and matched, to a predetermined folder

## stipulations
- jupyter notebook (for use as hugo site later)
- visualize/demonstrate as much as possible
- small gui to select images, crop, and select reference point
- lightweight, self-contained (conda managed)

## 🚀 quick start

### environment setup
```bash
# create conda environment
make setup
# or manually: conda env create -f environment.yml

# activate environment
conda activate nimslo_processing

# start jupyter lab
make lab
# or manually: jupyter lab
```

### processing workflow
1. open `processor.ipynb`
2. run all cells to load functions
3. use `process_nimslo_batch(processor)` for automated pipeline
4. or run step-by-step for more control

## 🐍 environment management

### local → remote sync
```bash
# after installing new packages locally
make export                    # export environment
git add environment.yml        # commit changes
git push                       # sync to remote

# on remote server
git pull                       # get latest environment
make update                    # update conda environment
# restart jupyter kernel
```

### available commands
- `make setup` - initial environment creation
- `make update` - update from environment.yml
- `make export` - export current environment
- `make verify` - test package installation
- `make lab` - start jupyter lab
- `make clean` - remove environment

see `environment_setup.md` for detailed workflow guide

