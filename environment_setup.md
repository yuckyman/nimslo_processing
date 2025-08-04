# conda environment sync guide 🐍

managing your nimslo processing environment across local machine and remote jupyter server.

## initial setup

### 1. create environment from file (both local + remote)

```bash
# create the environment
conda env create -f environment.yml

# activate it
conda activate nimslo_processing

# verify installation
python -c "import cv2, numpy, matplotlib; print('✅ core packages loaded')"
```

### 2. register jupyter kernel

```bash
# while environment is active
python -m ipykernel install --user --name nimslo_processing --display-name "nimslo processing"

# verify kernel is available
jupyter kernelspec list
```

## daily workflow

### updating packages

```bash
# activate environment
conda activate nimslo_processing

# install new package
conda install package_name
# or pip install package_name

# export updated environment
conda env export > environment.yml

# clean up the exported file (optional)
# remove the prefix: line and any pip hash values
```

### syncing between machines

**local → remote:**
```bash
# export current environment
conda env export --no-builds > environment.yml

# commit/push to git or scp to remote
git add environment.yml && git commit -m "update environment"
# or scp environment.yml user@remote:/path/to/project/
```

**remote update:**
```bash
# pull changes
git pull
# or scp from local

# update environment
conda env update -f environment.yml --prune

# restart jupyter kernel after major updates
```

## useful commands

```bash
# list environments
conda env list

# activate environment
conda activate nimslo_processing

# deactivate
conda deactivate

# remove environment (if needed)
conda env remove -n nimslo_processing

# export minimal environment (no builds/versions)
conda env export --no-builds > environment.yml

# export from history (what you explicitly installed)
conda env export --from-history > environment_minimal.yml
```

## troubleshooting

**jupyter kernel not showing up:**
```bash
conda activate nimslo_processing
python -m ipykernel install --user --name nimslo_processing --display-name "nimslo processing" --force
```

**package conflicts:**
```bash
# try conda-forge channel first
conda install -c conda-forge package_name

# or use mamba (faster solver)
mamba install package_name
```

**environment sync issues:**
```bash
# nuclear option: recreate environment
conda env remove -n nimslo_processing
conda env create -f environment.yml
```

## pro tips

- use `--no-builds` flag when exporting to avoid platform-specific issues
- commit environment.yml to git for version control
- use `conda env export --from-history` for minimal dependency files
- consider using `mamba` instead of `conda` for faster package resolution
- pin critical package versions if you need exact reproducibility