import copy
import traceback
import pandas as pd
import os

from .variance_unet_region_experiment import VertebraPatchExperiment


def run_experiment_grid(base_config: dict, experiment_grid: list) -> pd.DataFrame:
    all_results = []

    for i, exp_params in enumerate(experiment_grid, start=1):
        cfg = copy.deepcopy(base_config)

        model_type = exp_params["model_type"]
        patch_h, patch_w = exp_params["patch_size"]
        pad_x = exp_params["pad_x"]
        pad_y = exp_params["pad_y"]

        cfg["model"]["type"] = model_type
        cfg["execution_mode"] = model_type
        cfg["data"]["patch_size"] = [patch_h, patch_w]
        cfg["extractor"]["pad_x"] = pad_x
        cfg["extractor"]["pad_y"] = pad_y

        cfg["experiment_name"] = (
            f"{base_config['experiment_name']}"
            f"__model-{model_type}"
            f"__ps-{patch_h}x{patch_w}"
            f"__padx-{pad_x}"
            f"__pady-{pad_y}"
        )

        print("=" * 120)
        print(f"[{i}/{len(experiment_grid)}] Running: {cfg['experiment_name']}")

        try:
            experiment = VertebraPatchExperiment(cfg)
            result = experiment.fit()

            row = {
                "experiment_name": cfg["experiment_name"],
                "model_type": model_type,
                "patch_size": f"{patch_h}x{patch_w}",
                "pad_x": pad_x,
                "pad_y": pad_y,
                "best_epoch": result.get("best_epoch"),
                "best_metric_name": result.get("best_metric_name"),
                "best_metric_value": result.get("best_metric_value"),
                "test_acc": result.get("test_metrics", {}).get("acc"),
                "test_loss": result.get("test_metrics", {}).get("test_loss"),
                "output_dir": result.get("output_dir"),
                "status": result.get("status", "completed"),
            }
            all_results.append(row)

        except Exception as e:
            print(f"ERROR en {cfg['experiment_name']}: {e}")
            traceback.print_exc()

            all_results.append({
                "experiment_name": cfg["experiment_name"],
                "model_type": model_type,
                "patch_size": f"{patch_h}x{patch_w}",
                "pad_x": pad_x,
                "pad_y": pad_y,
                "status": "failed",
                "error": str(e),
            })

    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(base_config["drive_root"], "regions_experiment_summary.csv")
    results_df.to_csv(results_csv, index=False)

    print("Resumen guardado en:", results_csv)
    return results_df.sort_values(by=["test_acc"], ascending=False, na_position="last")
