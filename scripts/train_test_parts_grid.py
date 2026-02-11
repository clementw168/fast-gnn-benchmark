import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=False, default="configs/ogbn_products/sage.yml")
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 101112]
    parts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for train_part_idx, train_parts in enumerate(parts_list):
        for val_test_part_idx, val_test_parts in enumerate(parts_list):
            for seed_idx, seed in enumerate(seeds):
                print()
                print(f"Running for train_parts={train_parts} val_test_parts={val_test_parts} and seed={seed}")
                print(
                    f"({train_part_idx * len(seeds) * len(parts_list) + val_test_part_idx * len(seeds) + seed_idx + 1}/{len(parts_list) * len(seeds) * len(parts_list)})"
                )
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "src/fast_gnn_benchmark/main.py",
                        "--train_parts",
                        str(train_parts),
                        "--val_test_parts",
                        str(val_test_parts),
                        "--seed",
                        str(seed),
                        "--config_file",
                        args.config_file,
                        "--tag",
                        args.tag,
                    ],
                    check=False,
                )
                print(
                    f"Done for train_parts={train_parts} val_test_parts={val_test_parts} and seed={seed} ({train_part_idx * len(seeds) * len(parts_list) + val_test_part_idx * len(seeds) + seed_idx + 1}/{len(parts_list) * len(seeds) * len(parts_list)})"
                )
                print("________________________________________________________")
                print()
