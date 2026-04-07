import copy
import itertools
import traceback
import pandas as pd
import os

from .vertebra_variance_experiment_v2 import VertebraVarianceExperimentV2
from .discord_webhook_notifier import DiscordWebhookNotifier


def build_experiment_batch(
    model_types=None,
    patch_sizes=None,
    pad_xs=None,
    pad_ys=None,
    lrs=None,
    base_channels_list=None,
    batch_sizes=None,
    epochs_list=None,
    variance_ksizes=None,
    seeds=None,
    min_areas=None,
):
    model_types = model_types or ["baseline"]
    patch_sizes = patch_sizes or [(64, 64)]
    pad_xs = pad_xs or [30]
    pad_ys = pad_ys or [15]
    lrs = lrs or [1e-3]
    base_channels_list = base_channels_list or [32]
    batch_sizes = batch_sizes or [16]
    epochs_list = epochs_list or [20]
    variance_ksizes = variance_ksizes or [5]
    seeds = seeds or [42]
    min_areas = min_areas or [50]

    combos = list(itertools.product(
        model_types, patch_sizes, pad_xs, pad_ys, lrs,
        base_channels_list, batch_sizes, epochs_list,
        variance_ksizes, seeds, min_areas,
    ))

    batch = []
    for idx, (model_type, (ph, pw), px, py, lr, bch, bs, ep, vk, seed, ma) in enumerate(combos, start=1):
        batch.append({
            "id": idx,
            "model_type": model_type,
            "patch_size": [ph, pw],
            "pad_x": px,
            "pad_y": py,
            "lr": lr,
            "base_channels": bch,
            "batch_size": bs,
            "epochs": ep,
            "variance_ksize": vk,
            "seed": seed,
            "min_area": ma,
        })
    return batch


def run_experiment_grid(base_config: dict, experiment_batch: list) -> pd.DataFrame:
    all_results = []
    total = len(experiment_batch)

    webhook_url = base_config.get("discord_webhook_url", "")
    notifier = None
    if webhook_url:
        notifier = DiscordWebhookNotifier(webhook_url, experiment_name="GRID")
        notifier.send_grid_start(total, experiment_batch)

    for i, params in enumerate(experiment_batch, start=1):
        cfg = copy.deepcopy(base_config)

        exp_id = params["id"]
        model_type = params["model_type"]
        ph, pw = params["patch_size"]
        px = params["pad_x"]
        py = params["pad_y"]
        lr = params["lr"]
        bch = params.get("base_channels", 32)
        bs = params.get("batch_size", 16)
        ep = params.get("epochs", 20)
        vk = params.get("variance_ksize", 5)
        seed = params.get("seed", 42)
        ma = params.get("min_area", 50)

        cfg["model"]["type"] = model_type
        cfg["model"]["base_channels"] = bch
        cfg["data"]["patch_size"] = [ph, pw]
        cfg["extractor"]["pad_x"] = px
        cfg["extractor"]["pad_y"] = py
        cfg["extractor"]["min_area"] = ma
        cfg["training"]["lr"] = lr
        cfg["training"]["batch_size"] = bs
        cfg["training"]["epochs"] = ep
        cfg["seed"] = seed
        cfg["variance_ksize"] = vk

        cfg["experiment_name"] = (
            f"{base_config.get('experiment_name', 'exp')}"
            f"__{exp_id:03d}"
            f"__model-{model_type}"
            f"__ps-{ph}x{pw}"
            f"__padx-{px}"
            f"__pady-{py}"
            f"__lr-{lr}"
            f"__ch-{bch}"
            f"__bs-{bs}"
            f"__ep-{ep}"
            f"__vk-{vk}"
            f"__seed-{seed}"
        )

        print("=" * 120)
        print(f"[{i}/{total}] ID={exp_id}  {cfg['experiment_name']}")
        print("=" * 120)

        try:
            experiment = VertebraVarianceExperimentV2(cfg)
            result = experiment.fit()

            num_vert = result.get("num_vertebras", 0)

            row = {
                "id": exp_id,
                "experiment_name": cfg["experiment_name"],
                "model_type": model_type,
                "patch_size": f"{ph}x{pw}",
                "pad_x": px,
                "pad_y": py,
                "lr": lr,
                "base_channels": bch,
                "batch_size": bs,
                "epochs": ep,
                "variance_ksize": vk,
                "seed": seed,
                "min_area": ma,
                "num_vertebras": num_vert,
                "best_epoch": result.get("best_epoch"),
                "best_metric_name": result.get("best_metric_name"),
                "best_metric_value": result.get("best_metric_value"),
                "test_acc": result.get("test_metrics", {}).get("acc"),
                "test_loss": result.get("test_metrics", {}).get("test_loss"),
                "output_dir": result.get("output_dir"),
                "status": result.get("status", "completed"),
            }
            all_results.append(row)

            print(f"  ID={exp_id}  vertebras={num_vert}  val_acc={row['best_metric_value']:.4f}  test_acc={row['test_acc']:.4f}")

            if notifier:
                notifier.send_grid_row(row)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            row = {
                "id": exp_id,
                "experiment_name": cfg["experiment_name"],
                "model_type": model_type,
                "patch_size": f"{ph}x{pw}",
                "pad_x": px,
                "pad_y": py,
                "lr": lr,
                "base_channels": bch,
                "batch_size": bs,
                "epochs": ep,
                "variance_ksize": vk,
                "seed": seed,
                "min_area": ma,
                "status": "failed",
                "error": str(e),
            }
            all_results.append(row)

            if notifier:
                notifier.send_grid_row(row)

    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(base_config["drive_root"], "experiment_grid_summary.csv")
    results_df.to_csv(results_csv, index=False)

    ok = sum(1 for r in all_results if r.get("status") == "completed")
    fail = total - ok
    print(f"\nResumen guardado en: {results_csv}")
    print(f"Total: {total} | OK: {ok} | FAIL: {fail}")

    if notifier:
        notifier.send_grid_summary(results_df, results_csv)

    return results_df.sort_values(by=["test_acc"], ascending=False, na_position="last")
