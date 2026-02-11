import subprocess

if __name__ == "__main__":
    max_lr = 0.003
    max_lr_parts = 10
    min_lr = 0.001
    min_lr_parts = 1

    seeds = [42, 123, 456, 789, 101112]
    parts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for train_part_idx, train_parts in enumerate(parts_list):
        lr = min_lr + (max_lr - min_lr) * (train_parts - min_lr_parts) / (max_lr_parts - min_lr_parts)
        for seed_idx, seed in enumerate(seeds):
            print()
            print(f"Running for train_parts={train_parts} and seed={seed}")
            print(f"({train_part_idx * len(seeds) + seed_idx + 1}/{len(parts_list) * len(seeds)})")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "src/fast_gnn_benchmark/main.py",
                    "--config_file",
                    "configs/ogbn_products/sage_parts.yml",
                    "--train_parts",
                    str(train_parts),
                    "--val_test_parts",
                    str(1),
                    "--seed",
                    str(seed),
                    "--lr",
                    str(lr),
                    "--tag",
                    "products_train_parts_linear_lr",
                ],
                check=False,
            )
            print(
                f"Done for train_parts={train_parts} and seed={seed} ({train_part_idx * len(seeds) + seed_idx + 1}/{len(parts_list) * len(seeds)})"
            )
            print("________________________________________________________")
            print()
