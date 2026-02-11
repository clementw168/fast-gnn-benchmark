import subprocess

if __name__ == "__main__":
    datasets = ["ogbn_arxiv", "ogbn_products", "pokec"]
    seeds = [42, 123, 456, 789, 101112]
    parts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for dataset_idx, dataset in enumerate(datasets):
        for train_part_idx, train_parts in enumerate(parts_list):
            for seed_idx, seed in enumerate(seeds):
                print()
                print(f"Running for dataset={dataset} and train_parts={train_parts} and seed={seed}")
                print(
                    f"({dataset_idx * len(seeds) * len(parts_list) + train_part_idx * len(seeds) + seed_idx + 1}/{len(datasets) * len(seeds) * len(parts_list) * len(datasets)})"
                )
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "scripts/main.py",
                        "--config_file",
                        f"configs/{dataset}/sage_parts.yml",
                        "--train_parts",
                        str(train_parts),
                        "--val_test_parts",
                        str(1),
                        "--seed",
                        str(seed),
                        "--tag",
                        f"{dataset}_train_parts",
                    ],
                    check=False,
                )
                print(
                    f"Done for dataset={dataset} and train_parts={train_parts} and seed={seed} ({dataset_idx * len(seeds) * len(parts_list) + train_part_idx * len(seeds) + seed_idx + 1}/{len(datasets) * len(seeds) * len(parts_list) * len(datasets)})"
                )
                print("________________________________________________________")
                print()
