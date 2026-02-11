import subprocess

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 101112]
    config_files = [
        "configs/ogbn_arxiv/sage_parts.yml",
        "configs/ogbn_arxiv/sage_full.yml",
        "configs/ogbn_arxiv/gcn_parts.yml",
        "configs/ogbn_arxiv/gcn_full.yml",
        "configs/ogbn_arxiv/gat_parts.yml",
        "configs/ogbn_arxiv/gat_full.yml",
        "configs/ogbn_arxiv/sgformer_parts.yml",
        "configs/ogbn_arxiv/sgformer_full.yml",
        "configs/ogbn_products/sage_parts.yml",
        "configs/ogbn_products/sage_full.yml",
        "configs/ogbn_products/gcn_parts.yml",
        "configs/ogbn_products/gcn_full.yml",
        "configs/ogbn_products/gat_parts.yml",
        "configs/ogbn_products/gat_full.yml",
        "configs/ogbn_products/sgformer_parts.yml",
        "configs/ogbn_products/sgformer_full.yml",
        "configs/pokec/sage_parts.yml",
        "configs/pokec/sage_full.yml",
        "configs/pokec/gcn_parts.yml",
        "configs/pokec/gcn_full.yml",
        "configs/pokec/gat_parts.yml",
        "configs/pokec/gat_full.yml",
        "configs/pokec/sgformer_parts.yml",
        "configs/pokec/sgformer_full.yml",
    ]
    for config_file_idx, config_file in enumerate(config_files):
        for seed_idx, seed in enumerate(seeds):
            print()
            print(f"Running for config_file={config_file} and seed={seed}")
            print(f"({config_file_idx * len(seeds) + seed_idx + 1}/{len(config_files) * len(seeds)})")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "src/fast_gnn_benchmark/main.py",
                    "--config_file",
                    config_file,
                    "--seed",
                    str(seed),
                    "--tag",
                    "benchmark_architectures",
                ],
                check=False,
            )
            print(
                f"Done for config_file={config_file} and seed={seed} ({config_file_idx * len(seeds) + seed_idx + 1}/{len(config_files) * len(seeds)})"
            )
            print("________________________________________________________")
            print()
