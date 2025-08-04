# nimslo processing makefile
# convenient commands for environment management

.PHONY: setup update export verify clean help

# setup new environment
setup:
	@echo "🐍 setting up nimslo processing environment..."
	conda env create -f environment.yml
	conda run -n nimslo_processing python -m ipykernel install --user --name nimslo_processing --display-name "nimslo processing"
	@echo "✅ setup complete! activate with: conda activate nimslo_processing"

# update existing environment
update:
	@echo "📥 updating environment..."
	conda env update -f environment.yml --prune
	@echo "✅ environment updated!"

# export current environment
export:
	@echo "📤 exporting environment..."
	conda activate nimslo_processing && conda env export --no-builds > environment.yml
	@echo "✅ environment exported to environment.yml"

# verify environment works
verify:
	@echo "🔍 verifying environment..."
	./sync_env.sh verify

# clean up environment
clean:
	@echo "🧹 removing nimslo_processing environment..."
	conda env remove -n nimslo_processing -y
	jupyter kernelspec remove nimslo_processing -y || true
	@echo "✅ environment cleaned up"

# start jupyter lab
lab:
	@echo "🚀 starting jupyter lab..."
	conda activate nimslo_processing && jupyter lab

# show available commands
help:
	@echo "nimslo processing environment commands:"
	@echo ""
	@echo "  make setup    - create environment from environment.yml"
	@echo "  make update   - update existing environment"
	@echo "  make export   - export current environment"
	@echo "  make verify   - test environment packages"
	@echo "  make lab      - start jupyter lab"
	@echo "  make clean    - remove environment completely"
	@echo "  make help     - show this message"
	@echo ""
	@echo "workflow example:"
	@echo "  make setup           # initial setup"
	@echo "  make lab             # start working"
	@echo "  # ... install packages ..."
	@echo "  make export          # save environment"
	@echo "  # ... commit/push environment.yml ..."