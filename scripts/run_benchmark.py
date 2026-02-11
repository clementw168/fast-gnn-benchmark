import subprocess

datasets = ["pokec", "ogbn_products", "ogbn_arxiv"]
seeds = [42, 123, 456, 789, 101112]
config_files = ["sage_parts.yml", "sage_cluster.yml", "sage_full.yml", "sage_neighbour.yml", "sage_saint.yml"]

for dataset_idx, dataset in enumerate(datasets):
    for config_file_idx, config_file in enumerate(config_files):
        for seed_idx, seed in enumerate(seeds):
            subprocess.run(
                [
                    "uv",
                    "run",
                    "src/fast_gnn_benchmark/main.py",
                    "--config_file",
                    f"configs/{dataset}/{config_file}",
                    "--tag",
                    f"{dataset}_benchmark",
                    "--seed",
                    str(seed),
                ],
                check=False,
            )
            print(f"Done for {dataset} and {config_file} and seed={seed}")
            print(
                f"({dataset_idx * len(config_files) * len(seeds) + config_file_idx * len(seeds) + seed_idx + 1}/{len(datasets) * len(config_files) * len(seeds)})"
            )
            print("________________________________________________________")
            print()
