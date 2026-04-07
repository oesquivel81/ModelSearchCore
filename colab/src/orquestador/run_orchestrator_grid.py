import copy
import itertools
import time
import traceback
import os
import pandas as pd

from .vertebra_cnn_orchestrator import VertebraCNNOrchestrator
from src.utils.discord_webhook_notifier import DiscordWebhookNotifier


def build_orchestrator_batch(
    model_types=None,
    patch_sizes=None,
    subpatch_sizes=None,
    strides=None,
    pad_xs=None,
    pad_ys=None,
    lrs=None,
    base_channels_list=None,
    batch_sizes=None,
    epochs_list=None,
    variance_ksizes=None,
    seeds=None,
    min_areas=None,
    num_classes_list=None,
):
    model_types = model_types or ["baseline"]
    patch_sizes = patch_sizes or [(128, 128)]
    subpatch_sizes = subpatch_sizes or [(32, 32)]
    strides = strides or [(32, 32)]
    pad_xs = pad_xs or [30]
    pad_ys = pad_ys or [15]
    lrs = lrs or [1e-3]
    base_channels_list = base_channels_list or [32]
    batch_sizes = batch_sizes or [16]
    epochs_list = epochs_list or [20]
    variance_ksizes = variance_ksizes or [5]
    seeds = seeds or [42]
    min_areas = min_areas or [50]
    num_classes_list = num_classes_list or [13]

    combos = list(itertools.product(
        model_types, patch_sizes, subpatch_sizes, strides,
        pad_xs, pad_ys, lrs, base_channels_list,
        batch_sizes, epochs_list, variance_ksizes,
        seeds, min_areas, num_classes_list,
    ))

    batch = []
    for idx, (
        model_type, (ph, pw), (sh, sw), (strh, strw),
        px, py, lr, bch, bs, ep, vk, seed, ma, nc,
    ) in enumerate(combos, start=1):
        batch.append({
            "id": idx,
            "model_type": model_type,
            "patch_size": [ph, pw],
            "subpatch_size": [sh, sw],
            "stride": [strh, strw],
            "pad_x": px,
            "pad_y": py,
            "lr": lr,
            "base_channels": bch,
            "batch_size": bs,
            "epochs": ep,
            "variance_ksize": vk,
            "seed": seed,
            "min_area": ma,
            "num_classes": nc,
        })
    return batch


def _apply_params_to_config(base_config, params):
    cfg = copy.deepcopy(base_config)

    exp_id = params["id"]
    model_type = params["model_type"]
    ph, pw = params["patch_size"]
    sh, sw = params["subpatch_size"]
    strh, strw = params["stride"]
    px = params["pad_x"]
    py = params["pad_y"]
    lr = params["lr"]
    bch = params["base_channels"]
    bs = params["batch_size"]
    ep = params["epochs"]
    vk = params["variance_ksize"]
    seed = params["seed"]
    ma = params["min_area"]
    nc = params["num_classes"]

    cfg["seed"] = seed
    cfg.setdefault("model", {})
    cfg["model"]["type"] = model_type
    cfg["model"]["base_channels"] = bch
    cfg["model"]["num_classes"] = nc

    cfg.setdefault("data", {})
    cfg["data"]["patch_size"] = [ph, pw]

    cfg.setdefault("subpatches", {})
    cfg["subpatches"]["subpatch_size"] = [sh, sw]
    cfg["subpatches"]["stride"] = [strh, strw]

    cfg.setdefault("extractor", {})
    cfg["extractor"]["pad_x"] = px
    cfg["extractor"]["pad_y"] = py
    cfg["extractor"]["min_area"] = ma

    cfg.setdefault("variance", {})
    cfg["variance"]["variance_ksize"] = vk

    cfg.setdefault("training", {})
    cfg["training"]["lr"] = lr
    cfg["training"]["batch_size"] = bs
    cfg["training"]["epochs"] = ep

    cfg["experiment_name"] = (
        f"{base_config.get('experiment_name', 'orch')}"
        f"__{exp_id:03d}"
        f"__model-{model_type}"
        f"__ps-{ph}x{pw}"
        f"__sp-{sh}x{sw}"
        f"__padx-{px}"
        f"__lr-{lr}"
        f"__ch-{bch}"
        f"__bs-{bs}"
        f"__ep-{ep}"
        f"__vk-{vk}"
        f"__seed-{seed}"
    )

    return cfg


def run_orchestrator_grid(base_config: dict, experiment_batch: list) -> pd.DataFrame:
    all_results = []
    total = len(experiment_batch)
    grid_t0 = time.time()

    def _grid_log(msg):
        elapsed = time.time() - grid_t0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print(f"[{h:02d}:{m:02d}:{s:02d}] [GRID] {msg}", flush=True)

    _grid_log(f"Iniciando grid con {total} experimentos")

    webhook_url = base_config.get("discord", {}).get("webhook_url", "")
    notifier = None
    if webhook_url:
        notifier = DiscordWebhookNotifier(webhook_url, experiment_name="ORCH_GRID")
        notifier.send_grid_start(total, experiment_batch)

    for i, params in enumerate(experiment_batch, start=1):
        cfg = _apply_params_to_config(base_config, params)
        exp_id = params["id"]

        _grid_log("=" * 80)
        _grid_log(f"Experimento [{i}/{total}] ID={exp_id}  {cfg['experiment_name']}")
        _grid_log("=" * 80)
        exp_t0 = time.time()

        try:
            orch = VertebraCNNOrchestrator(cfg)
            result = orch.fit()
            exp_min = (time.time() - exp_t0) / 60

            row = {
                "id": exp_id,
                "experiment_name": cfg["experiment_name"],
                "model_type": params["model_type"],
                "patch_size": f"{params['patch_size'][0]}x{params['patch_size'][1]}",
                "subpatch_size": f"{params['subpatch_size'][0]}x{params['subpatch_size'][1]}",
                "pad_x": params["pad_x"],
                "pad_y": params["pad_y"],
                "lr": params["lr"],
                "base_channels": params["base_channels"],
                "batch_size": params["batch_size"],
                "epochs": params["epochs"],
                "variance_ksize": params["variance_ksize"],
                "seed": params["seed"],
                "min_area": params["min_area"],
                "num_classes": params["num_classes"],
                "num_studies": result.get("num_studies"),
                "num_vertebra_regions": result.get("num_vertebra_regions"),
                "num_subpatches": result.get("num_subpatches"),
                "best_epoch": result.get("best_epoch"),
                "best_metric_name": result.get("best_metric_name"),
                "best_metric_value": result.get("best_metric_value"),
                "test_acc": result.get("test_metrics", {}).get("acc"),
                "test_f1_macro": result.get("test_metrics", {}).get("f1_macro"),
                "test_loss": result.get("test_metrics", {}).get("loss"),
                "output_dir": result.get("output_dir"),
                "status": "completed",
            }
            all_results.append(row)

            _grid_log(
                f"\u2705 ID={exp_id}  {exp_min:.1f} min | "
                f"subpatches={row['num_subpatches']}  "
                f"val_{row['best_metric_name']}={row['best_metric_value']:.4f}  "
                f"test_acc={row['test_acc']:.4f}  test_f1={row['test_f1_macro']:.4f}"
            )

            if notifier:
                notifier.send_grid_row(row)

        except Exception as e:
            exp_min = (time.time() - exp_t0) / 60
            _grid_log(f"\u274c ID={exp_id}  {exp_min:.1f} min | ERROR: {e}")
            traceback.print_exc()
            row = {
                "id": exp_id,
                "experiment_name": cfg["experiment_name"],
                "model_type": params["model_type"],
                "patch_size": f"{params['patch_size'][0]}x{params['patch_size'][1]}",
                "subpatch_size": f"{params['subpatch_size'][0]}x{params['subpatch_size'][1]}",
                "pad_x": params["pad_x"],
                "pad_y": params["pad_y"],
                "lr": params["lr"],
                "base_channels": params["base_channels"],
                "batch_size": params["batch_size"],
                "epochs": params["epochs"],
                "variance_ksize": params["variance_ksize"],
                "seed": params["seed"],
                "min_area": params["min_area"],
                "num_classes": params["num_classes"],
                "status": "failed",
                "error": str(e),
            }
            all_results.append(row)

            if notifier:
                notifier.send_grid_row(row)

    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(base_config["drive_root"], "orchestrator_grid_summary.csv")
    results_df.to_csv(results_csv, index=False)

    ok = sum(1 for r in all_results if r.get("status") == "completed")
    fail = total - ok
    total_min = (time.time() - grid_t0) / 60
    _grid_log(f"Grid finalizado en {total_min:.1f} min | OK: {ok}/{total} | FAIL: {fail}")
    _grid_log(f"CSV: {results_csv}")

    if notifier:
        notifier.send_grid_summary(results_df, results_csv)

    return results_df.sort_values(by=["test_f1_macro", "test_acc"], ascending=False, na_position="last")
