
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import os
import numpy as np
import pandas as pd


class PreTDAMetricsBuilder:
	"""
	Calcula métricas previas a TDA a nivel:
	- región/parche
	- ventana consecutiva
	- resumen por filtro/config/paciente

	Supuestos:
	- Cada región puede venir como dataclass, objeto o dict.
	- Cada región puede tener campos como:
		region_id, image_path, patient_id, config_id, filter_name,
		centroid, order_index, vertebra_idx, curve_param, bbox, metadata,
		mean_dice, mean_iou, mean_mse_img, mean_mae_img,
		mean_grad_mse, mean_grad_mae, mean_var_diff, mean_intensity_diff
	- Las ventanas son listas de regiones ya ordenadas espacialmente.
	"""

	DEFAULT_METRIC_COLUMNS = [
		"mean_dice",
		"mean_iou",
		"mean_mse_img",
		"mean_mae_img",
		"mean_grad_mse",
		"mean_grad_mae",
		"mean_var_diff",
		"mean_intensity_diff",
	]

	def __init__(
		self,
		metric_columns: Optional[Sequence[str]] = None,
		selection_mode: str = "spatial_consecutive_windows",
		ordering_source: str = "vertebra_idx",
		spatial_file_used: Optional[str] = None,
	) -> None:
		self.metric_columns = list(metric_columns or self.DEFAULT_METRIC_COLUMNS)
		self.selection_mode = selection_mode
		self.ordering_source = ordering_source
		self.spatial_file_used = spatial_file_used

	# =========================================================
	# Helpers
	# =========================================================
	def _to_dict(self, obj: Any) -> Dict[str, Any]:
		if obj is None:
			return {}
		if isinstance(obj, dict):
			return dict(obj)
		if is_dataclass(obj):
			return asdict(obj)
		if hasattr(obj, "__dict__"):
			return dict(vars(obj))
		return {}

	def _get(self, obj: Any, key: str, default: Any = None) -> Any:
		if isinstance(obj, dict):
			return obj.get(key, default)
		return getattr(obj, key, default)

	def _safe_float(self, value: Any) -> float:
		if value is None:
			return np.nan
		try:
			if isinstance(value, str) and not value.strip():
				return np.nan
			return float(value)
		except Exception:
			return np.nan

	def _parse_centroid(self, region: Any) -> Tuple[float, float]:
		centroid = self._get(region, "centroid", None)

		if isinstance(centroid, (tuple, list)) and len(centroid) >= 2:
			return self._safe_float(centroid[0]), self._safe_float(centroid[1])

		cx = self._get(region, "centroid_x", None)
		cy = self._get(region, "centroid_y", None)
		return self._safe_float(cx), self._safe_float(cy)

	def _nanmean(self, values: Sequence[float]) -> float:
		arr = np.array(values, dtype=float)
		return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

	def _nanmedian(self, values: Sequence[float]) -> float:
		arr = np.array(values, dtype=float)
		return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan

	def _nanstd(self, values: Sequence[float]) -> float:
		arr = np.array(values, dtype=float)
		return float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan

	def _nanmin(self, values: Sequence[float]) -> float:
		arr = np.array(values, dtype=float)
		return float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan

	def _nanmax(self, values: Sequence[float]) -> float:
		arr = np.array(values, dtype=float)
		return float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan

	def _nanrange(self, values: Sequence[float]) -> float:
		min_v = self._nanmin(values)
		max_v = self._nanmax(values)
		if np.isnan(min_v) or np.isnan(max_v):
			return np.nan
		return float(max_v - min_v)

	def _euclidean(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
		if any(np.isnan(v) for v in [p1[0], p1[1], p2[0], p2[1]]):
			return np.nan
		return float(math.dist(p1, p2))

	def _consecutive_abs_diffs(self, values: Sequence[float]) -> List[float]:
		diffs: List[float] = []
		for i in range(len(values) - 1):
			a, b = values[i], values[i + 1]
			if np.isnan(a) or np.isnan(b):
				diffs.append(np.nan)
			else:
				diffs.append(abs(a - b))
		return diffs

	def _metric_values_from_regions(self, regions: Sequence[Any], metric_name: str) -> List[float]:
		return [self._safe_float(self._get(r, metric_name, None)) for r in regions]

	# =========================================================
	# Región / parche
	# =========================================================
	def build_region_row(self, region: Any) -> Dict[str, Any]:
		cx, cy = self._parse_centroid(region)
		metadata = self._get(region, "metadata", {}) or {}

		row: Dict[str, Any] = {
			"record_level": "region",
			"region_id": self._get(region, "region_id"),
			"image_path": self._get(region, "image_path"),
			"patient_id": self._get(region, "patient_id"),
			"config_id": self._get(region, "config_id"),
			"filter_name": self._get(region, "filter_name"),
			"order_index": self._get(region, "order_index"),
			"vertebra_idx": self._get(region, "vertebra_idx"),
			"curve_param": self._get(region, "curve_param"),
			"centroid_x": cx,
			"centroid_y": cy,
			"bbox": self._get(region, "bbox"),
			"split": metadata.get("split"),
			"selection_mode": self.selection_mode,
			"ordering_source": self.ordering_source,
			"spatial_file_used": metadata.get("spatial_file_used", self.spatial_file_used),
		}

		for metric in self.metric_columns:
			row[metric] = self._safe_float(self._get(region, metric, None))

		return row

	def build_regions_dataframe(self, regions: Sequence[Any]) -> pd.DataFrame:
		rows = [self.build_region_row(r) for r in regions]
		return pd.DataFrame(rows)

	# =========================================================
	# Ventanas / combinaciones
	# =========================================================
	def build_window_row(
		self,
		regions: Sequence[Any],
		combination_id: Optional[str] = None,
		is_valid_simplex: Optional[bool] = None,
		validity_reason: Optional[str] = None,
		rejection_reason: Optional[str] = None,
	) -> Dict[str, Any]:
		if not regions:
			raise ValueError("regions no puede estar vacío")

		first = regions[0]
		k = len(regions)
		member_region_ids = [self._get(r, "region_id") for r in regions]
		member_image_paths = [self._get(r, "image_path") for r in regions]
		member_indices = [self._get(r, "order_index") for r in regions]
		member_centroids = [self._parse_centroid(r) for r in regions]

		xs = [c[0] for c in member_centroids]
		ys = [c[1] for c in member_centroids]

		centroid_steps = [
			self._euclidean(member_centroids[i], member_centroids[i + 1])
			for i in range(len(member_centroids) - 1)
		]

		row: Dict[str, Any] = {
			"record_level": "window",
			"combination_id": combination_id or "|".join(str(x) for x in member_region_ids),
			"patient_id": self._get(first, "patient_id"),
			"config_id": self._get(first, "config_id"),
			"filter_name": self._get(first, "filter_name"),
			"k": k,
			"simplex_dim": k - 1,
			"member_region_ids": member_region_ids,
			"member_image_paths": member_image_paths,
			"member_indices": member_indices,
			"member_centroids": member_centroids,
			"centroid_span_x": self._nanrange(xs),
			"centroid_span_y": self._nanrange(ys),
			"mean_centroid_distance": self._nanmean(centroid_steps),
			"max_centroid_distance": self._nanmax(centroid_steps),
			"window_mean_centroid_step": self._nanmean(centroid_steps),
			"window_max_centroid_step": self._nanmax(centroid_steps),
			"is_contiguous_in_order": self._is_contiguous(member_indices),
			"is_curve_consistent": True,  # cascarón; ajustar después si agregas regla real
			"is_valid_simplex": is_valid_simplex,
			"validity_reason": validity_reason,
			"rejection_reason": rejection_reason,
			"selection_mode": self.selection_mode,
			"ordering_source": self.ordering_source,
			"spatial_file_used": self._get(first, "metadata", {}).get("spatial_file_used", self.spatial_file_used),
		}

		# métricas agregadas por ventana
		for metric in self.metric_columns:
			vals = self._metric_values_from_regions(regions, metric)

			row[f"window_{metric}_mean"] = self._nanmean(vals)
			row[f"window_{metric}_median"] = self._nanmedian(vals)
			row[f"window_{metric}_std"] = self._nanstd(vals)
			row[f"window_{metric}_min"] = self._nanmin(vals)
			row[f"window_{metric}_max"] = self._nanmax(vals)
			row[f"window_{metric}_range"] = self._nanrange(vals)

			diffs = self._consecutive_abs_diffs(vals)
			row[f"window_mean_abs_diff_{metric}"] = self._nanmean(diffs)
			row[f"window_max_abs_diff_{metric}"] = self._nanmax(diffs)

		return row

	def build_windows_dataframe(
		self,
		windows: Sequence[Sequence[Any]],
		validity_records: Optional[Sequence[Dict[str, Any]]] = None,
	) -> pd.DataFrame:
		rows: List[Dict[str, Any]] = []

		if validity_records is None:
			validity_records = [{} for _ in windows]

		for regions, meta in zip(windows, validity_records):
			row = self.build_window_row(
				regions=regions,
				combination_id=meta.get("combination_id"),
				is_valid_simplex=meta.get("is_valid_simplex"),
				validity_reason=meta.get("validity_reason"),
				rejection_reason=meta.get("rejection_reason"),
			)
			rows.append(row)

		return pd.DataFrame(rows)

	def _is_contiguous(self, member_indices: Sequence[Any]) -> bool:
		clean = [x for x in member_indices if x is not None and not (isinstance(x, float) and np.isnan(x))]
		if len(clean) != len(member_indices):
			return False
		try:
			ints = [int(x) for x in clean]
		except Exception:
			return False

		return all(ints[i + 1] - ints[i] == 1 for i in range(len(ints) - 1))

	# =========================================================
	# Resumen
	# =========================================================
	def build_summary_dataframe(
		self,
		regions_df: pd.DataFrame,
		windows_df: pd.DataFrame,
	) -> pd.DataFrame:
		group_cols = ["patient_id", "config_id", "filter_name", "selection_mode", "ordering_source"]

		for col in group_cols:
			if col not in regions_df.columns:
				regions_df[col] = None
			if col not in windows_df.columns:
				windows_df[col] = None

		region_summary = (
			regions_df.groupby(group_cols, dropna=False)
			.agg(
				n_regions=("region_id", "count"),
				n_unique_images=("image_path", "nunique"),
			)
			.reset_index()
		)

		window_summary = (
			windows_df.groupby(group_cols, dropna=False)
			.agg(
				n_windows=("combination_id", "count"),
				n_valid_simplex=("is_valid_simplex", lambda s: int(np.nansum(pd.Series(s).fillna(False).astype(bool)))),
				mean_k=("k", "mean"),
				mean_simplex_dim=("simplex_dim", "mean"),
			)
			.reset_index()
		)

		summary = pd.merge(region_summary, window_summary, on=group_cols, how="outer")

		# agrega medias globales de métricas por región
		for metric in self.metric_columns:
			if metric in regions_df.columns:
				metric_summary = (
					regions_df.groupby(group_cols, dropna=False)[metric]
					.mean()
					.reset_index()
					.rename(columns={metric: f"region_{metric}_mean"})
				)
				summary = pd.merge(summary, metric_summary, on=group_cols, how="left")

			window_metric = f"window_{metric}_mean"
			if window_metric in windows_df.columns:
				metric_summary = (
					windows_df.groupby(group_cols, dropna=False)[window_metric]
					.mean()
					.reset_index()
					.rename(columns={window_metric: f"{window_metric}_global_mean"})
				)
				summary = pd.merge(summary, metric_summary, on=group_cols, how="left")

		return summary

	# =========================================================
	# CSV maestro
	# =========================================================
	def build_master_dataframe(
		self,
		regions_df: pd.DataFrame,
		windows_df: pd.DataFrame,
	) -> pd.DataFrame:
		common_cols = sorted(set(regions_df.columns).union(set(windows_df.columns)))
		regions_aligned = regions_df.reindex(columns=common_cols)
		windows_aligned = windows_df.reindex(columns=common_cols)
		return pd.concat([regions_aligned, windows_aligned], ignore_index=True)

	# =========================================================
	# Exportación
	# =========================================================
	def export_all(
		self,
		output_dir: str,
		regions_df: pd.DataFrame,
		windows_df: pd.DataFrame,
		summary_df: Optional[pd.DataFrame] = None,
		prefix: str = "pre_tda",
	) -> Dict[str, str]:
		os.makedirs(output_dir, exist_ok=True)

		if summary_df is None:
			summary_df = self.build_summary_dataframe(regions_df, windows_df)

		master_df = self.build_master_dataframe(regions_df, windows_df)

		paths = {
			"regions_report": os.path.join(output_dir, f"{prefix}_regions_report.csv"),
			"windows_report": os.path.join(output_dir, f"{prefix}_windows_report.csv"),
			"summary_report": os.path.join(output_dir, f"{prefix}_summary_report.csv"),
			"master_report": os.path.join(output_dir, f"{prefix}_master_table.csv"),
		}

		regions_df.to_csv(paths["regions_report"], index=False)
		windows_df.to_csv(paths["windows_report"], index=False)
		summary_df.to_csv(paths["summary_report"], index=False)
		master_df.to_csv(paths["master_report"], index=False)

		return paths
