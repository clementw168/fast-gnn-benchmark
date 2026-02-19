setup-linux:
	sed -i -E '/"torch-(geometric|scatter|sparse)[^"]*",/d' pyproject.toml
	uv sync &&\
	uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html &&\
	uv add torch-geometric

resetup-linux:
	rm -r .venv/ &&\
	rm uv.lock &&\
	uv clean cache &&\
	make setup-linux

jupyter:
	@printf 'from local, use this command to access the jupyter notebook: ssh -N -L 8888:localhost:8888 %s@%s\n' "$$(whoami)" "$$(hostname -s)"
	uv run jupyter notebook --no-browser --port=8888

cluster-info:
	@bash -lc 'set -euo pipefail; \
	sinfo -h -o %P | sed "s/\*$$//" | sort -u | \
	while IFS= read -r p; do \
		sum=0; \
		while IFS= read -r n; do \
			line="$$(scontrol show node -o "$$n")"; \
			cfg="$$(sed -n "s/.*CfgTRES=[^ ]*gres\/gpu=\([0-9]\+\).*/\1/p" <<<"$$line")"; \
			alloc="$$(sed -n "s/.*AllocTRES=[^ ]*gres\/gpu=\([0-9]\+\).*/\1/p" <<<"$$line")"; \
			cfg="$${cfg:-0}"; \
			alloc="$${alloc:-0}"; \
			sum="$$((sum + cfg - alloc))"; \
		done < <(sinfo -h -p "$$p" -N -o %N); \
		printf "%-12s free_gpus=%d\n" "$$p" "$$sum"; \
	done'