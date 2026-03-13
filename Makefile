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

spawn-on-gpu-gateway:
	ssh -t clwang@gpu-gw 'cd ~/fast-gnn-benchmark && make cluster-info && exec bash -l'

spawn-auto-cluster:
	ssh -t clwang@gpu-gw 'cd ~/fast-gnn-benchmark && make get-auto-cluster'

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

get-cluster:
	@test -n "$(CLUSTER)" || (echo "Usage: make get_cluster CLUSTER=cluster_name"; exit 1)
	sinteractive -p $(CLUSTER) -c 32 --mem 150G --time 10:00:00

get-auto-cluster:
	@bash -lc 'set -euo pipefail; \
	out="$$(make -s cluster-info)"; \
	printf "%s\n" "$$out"; \
	get_free() { \
	  part="$$1"; \
	  printf "%s\n" "$$out" | awk -v p="$$part" '\''$$1 == p { split($$2,a,"="); print a[2]+0; found=1 } END { if (!found) print 0 }'\''; \
	}; \
	for p in L40S audible A100 A40; do \
	  free="$$(get_free "$$p")"; \
	  echo "Checking $$p: free_gpus=$$free"; \
	  if [ "$$free" -gt 0 ]; then \
	    echo "Selecting cluster: $$p"; \
	    $(MAKE) get-cluster CLUSTER="$$p"; \
	    exit 0; \
	  fi; \
	done; \
	echo "No available cluster found in: L40S, audible, A100, A40"; \
	exit 1'

run-sweep:
	@test -n "$(SWEEP_ID)" || (echo "Usage: make run_sweep SWEEP_ID=sweep_id"; exit 1)
	sbatch --export=SWEEP_ID=$(SWEEP_ID) slurm/sweep_agent.sh

run-config:
	@test -n "$(CONFIG_FILE)" || (echo "Usage: make run_config CONFIG_FILE=path/to/config.yml"; exit 1)
	sbatch --export=CONFIG_FILE=$(CONFIG_FILE) slurm/run_config.sh