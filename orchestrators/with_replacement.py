import subprocess

if __name__ == "__main__":
    datasets = ["ogbn_arxiv", "pokec", "ogbn_products"]
    seeds = [42, 123, 456, 789, 101112]
    for dataset_idx, dataset in enumerate(datasets):
        for seed_idx, seed in enumerate(seeds):
            print(f"Running for seed={seed}")
            print(f"({dataset_idx * len(seeds) + seed_idx + 1}/{len(datasets) * len(seeds)})")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "scripts/main.py",
                    "--config_file",
                    f"configs/{dataset}/sage_replacement.yml",
                    "--seed",
                    str(seed),
                    "--tag",
                    f"with_replacement_{dataset}",
                ],
                check=False,
            )
            print(
                f"Done for dataset={dataset} and seed={seed} ({dataset_idx * len(seeds) + seed_idx + 1}/{len(datasets) * len(seeds)})"
            )
            print("________________________________________________________")
            print()
