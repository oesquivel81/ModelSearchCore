import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class VertebraComponentExtractor:
    def __init__(
        self,
        image,
        local_mask,
        min_area=150,
        pad_x=20,
        pad_y=15,
        top_region_ratio=0.30,
        top_pad_x_scale=1.25,
        top_pad_y_top_scale=2.2,
        top_pad_y_bottom_scale=0.8,
        save_dir=None
    ):
        self.image = image
        self.local_mask = local_mask
        self.min_area = min_area
        self.pad_x = pad_x
        self.pad_y = pad_y

        self.top_region_ratio = top_region_ratio
        self.top_pad_x_scale = top_pad_x_scale
        self.top_pad_y_top_scale = top_pad_y_top_scale
        self.top_pad_y_bottom_scale = top_pad_y_bottom_scale

        self.save_dir = save_dir

        self.components = []
        self.overlay = None

    def run(self):
        self._extract_components()
        self._build_overlay()
        return self

    def _extract_components(self):
        binary = (self.local_mask > 0).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        h, w = binary.shape
        comps = []

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])

            cx, cy = centroids[i]

            pad_x_left = self.pad_x
            pad_x_right = self.pad_x
            pad_y_top = self.pad_y
            pad_y_bottom = self.pad_y

            if cy < self.top_region_ratio * h:
                pad_x_left = int(round(self.pad_x * self.top_pad_x_scale))
                pad_x_right = int(round(self.pad_x * self.top_pad_x_scale))
                pad_y_top = int(round(self.pad_y * self.top_pad_y_top_scale))
                pad_y_bottom = int(round(self.pad_y * self.top_pad_y_bottom_scale))

            x1 = max(0, x - pad_x_left)
            y1 = max(0, y - pad_y_top)
            x2 = min(w, x + ww + pad_x_right)
            y2 = min(h, y + hh + pad_y_bottom)

            patch_img = self.image[y1:y2, x1:x2]
            patch_mask = self.local_mask[y1:y2, x1:x2]

            comps.append({
                "label": i,
                "area": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "bbox": (x1, y1, x2, y2),
                "patch_img": patch_img,
                "patch_mask": patch_mask
            })

        comps = sorted(comps, key=lambda d: d["centroid_y"])
        self.components = comps

    def _build_overlay(self):
        out = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        for idx, comp in enumerate(self.components):
            x1, y1, x2, y2 = comp["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 80), 2)

            cx = int(round(comp["centroid_x"]))
            cy = int(round(comp["centroid_y"]))
            cv2.circle(out, (cx, cy), 3, (255, 255, 0), -1)

            cv2.putText(
                out,
                str(idx),
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        self.overlay = out

    def show_overlay(self, figsize=(8, 12)):
        plt.figure(figsize=figsize)
        plt.imshow(self.overlay)
        plt.title("Componentes candidatas a vértebras")
        plt.axis("off")
        plt.show()

    def show_patches(self, max_patches=16, figsize=(16, 10)):
        n = min(max_patches, len(self.components))
        if n == 0:
            print("No hay componentes.")
            return

        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for i in range(n):
            comp = self.components[i]
            axes[i].imshow(comp["patch_img"], cmap="gray")
            axes[i].set_title(
                f"id={i}\ny={comp['centroid_y']:.1f} | area={comp['area']}"
            )
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def save_patches(self):
        if self.save_dir is None:
            raise ValueError("Define save_dir para guardar patches.")
        os.makedirs(self.save_dir, exist_ok=True)

        for i, comp in enumerate(self.components):
            img_path = os.path.join(self.save_dir, f"vertebra_{i:02d}.png")
            mask_path = os.path.join(self.save_dir, f"vertebra_{i:02d}_mask.png")
            cv2.imwrite(img_path, comp["patch_img"])
            cv2.imwrite(mask_path, comp["patch_mask"])

        overlay_path = os.path.join(self.save_dir, "overlay_components.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(self.overlay, cv2.COLOR_RGB2BGR))

        print(f"Guardado en: {self.save_dir}")

    # =========================================================
    # MÉTRICAS ENTRE CAJAS
    # =========================================================
    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _bbox_intersection(self, bbox1, bbox2):
        x1a, y1a, x2a, y2a = bbox1
        x1b, y1b, x2b, y2b = bbox2

        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b)
        yi2 = min(y2a, y2b)

        if xi2 <= xi1 or yi2 <= yi1:
            return None

        return (xi1, yi1, xi2, yi2)

    def _bbox_intersection_area(self, bbox1, bbox2):
        inter = self._bbox_intersection(bbox1, bbox2)
        if inter is None:
            return 0
        return self._bbox_area(inter)

    def _bbox_union_area(self, bbox1, bbox2):
        a1 = self._bbox_area(bbox1)
        a2 = self._bbox_area(bbox2)
        ai = self._bbox_intersection_area(bbox1, bbox2)
        return a1 + a2 - ai

    def _bbox_iou(self, bbox1, bbox2):
        ai = self._bbox_intersection_area(bbox1, bbox2)
        au = self._bbox_union_area(bbox1, bbox2)
        if au == 0:
            return 0.0
        return ai / au

    def _bbox_overlaps(self, bbox1, bbox2):
        return self._bbox_intersection(bbox1, bbox2) is not None

    def _bbox_min_distance(self, bbox1, bbox2):
        x1a, y1a, x2a, y2a = bbox1
        x1b, y1b, x2b, y2b = bbox2

        dx = max(x1b - x2a, x1a - x2b, 0)
        dy = max(y1b - y2a, y1a - y2b, 0)

        return float(np.sqrt(dx * dx + dy * dy))

    def _bbox_corners(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)

    def _bbox_border_points(self, bbox):
        x1, y1, x2, y2 = bbox

        pts = []

        for x in range(x1, x2 + 1):
            pts.append((x, y1))
            pts.append((x, y2))

        for y in range(y1, y2 + 1):
            pts.append((x1, y))
            pts.append((x2, y))

        pts = np.unique(np.array(pts, dtype=np.float32), axis=0)
        return pts

    def _directed_hausdorff_points(self, A, B):
        if len(A) == 0 or len(B) == 0:
            return float("inf")

        max_min_dist = 0.0
        for a in A:
            dists = np.sqrt(np.sum((B - a) ** 2, axis=1))
            min_dist = float(np.min(dists))
            if min_dist > max_min_dist:
                max_min_dist = min_dist
        return max_min_dist

    def _hausdorff_points(self, A, B):
        if len(A) == 0 and len(B) == 0:
            return 0.0
        if len(A) == 0 or len(B) == 0:
            return float("inf")

        dab = self._directed_hausdorff_points(A, B)
        dba = self._directed_hausdorff_points(B, A)
        return max(dab, dba)

    def _bbox_hausdorff_corners(self, bbox1, bbox2):
        A = self._bbox_corners(bbox1)
        B = self._bbox_corners(bbox2)
        return self._hausdorff_points(A, B)

    def _bbox_hausdorff_border(self, bbox1, bbox2):
        A = self._bbox_border_points(bbox1)
        B = self._bbox_border_points(bbox2)
        return self._hausdorff_points(A, B)

    def bbox_pair_metrics(self, idx1, idx2):
        comp1 = self.components[idx1]
        comp2 = self.components[idx2]

        bbox1 = comp1["bbox"]
        bbox2 = comp2["bbox"]

        inter = self._bbox_intersection(bbox1, bbox2)
        inter_area = self._bbox_intersection_area(bbox1, bbox2)
        union_area = self._bbox_union_area(bbox1, bbox2)
        iou = self._bbox_iou(bbox1, bbox2)
        min_dist = self._bbox_min_distance(bbox1, bbox2)
        h_corners = self._bbox_hausdorff_corners(bbox1, bbox2)
        h_border = self._bbox_hausdorff_border(bbox1, bbox2)

        cx1, cy1 = comp1["centroid_x"], comp1["centroid_y"]
        cx2, cy2 = comp2["centroid_x"], comp2["centroid_y"]

        centroid_dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))
        delta_y = abs(cy1 - cy2)
        delta_x = abs(cx1 - cx2)

        return {
            "idx1": idx1,
            "idx2": idx2,
            "bbox1": bbox1,
            "bbox2": bbox2,
            "overlap": inter is not None,
            "intersection_bbox": inter,
            "intersection_area": int(inter_area),
            "union_area": int(union_area),
            "iou": float(iou),
            "min_distance": float(min_dist),
            "centroid_distance": float(centroid_dist),
            "delta_x_centroid": float(delta_x),
            "delta_y_centroid": float(delta_y),
            "hausdorff_corners": float(h_corners),
            "hausdorff_border": float(h_border),
        }

    def all_bbox_pair_metrics(self, only_consecutive=False):
        results = []
        n = len(self.components)

        if only_consecutive:
            for i in range(n - 1):
                results.append(self.bbox_pair_metrics(i, i + 1))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    results.append(self.bbox_pair_metrics(i, j))

        return results

    def find_overlapping_or_close_pairs(
        self,
        iou_threshold=0.05,
        min_distance_threshold=15.0,
        hausdorff_border_threshold=25.0,
        only_consecutive=True
    ):
        pairs = self.all_bbox_pair_metrics(only_consecutive=only_consecutive)

        selected = []
        for m in pairs:
            if (
                m["iou"] > iou_threshold
                or m["min_distance"] < min_distance_threshold
                or m["hausdorff_border"] < hausdorff_border_threshold
            ):
                selected.append(m)

        return selected

    def print_pair_metrics(self, only_consecutive=True, max_rows=50):
        pairs = self.all_bbox_pair_metrics(only_consecutive=only_consecutive)

        print("-" * 120)
        print(
            f"{'i':>3} {'j':>3} | {'IoU':>8} | {'min_dist':>9} | "
            f"{'cent_dist':>10} | {'dY':>8} | {'HausB':>8} | {'overlap':>8}"
        )
        print("-" * 120)

        for k, m in enumerate(pairs[:max_rows]):
            print(
                f"{m['idx1']:>3} {m['idx2']:>3} | "
                f"{m['iou']:>8.4f} | "
                f"{m['min_distance']:>9.2f} | "
                f"{m['centroid_distance']:>10.2f} | "
                f"{m['delta_y_centroid']:>8.2f} | "
                f"{m['hausdorff_border']:>8.2f} | "
                f"{str(m['overlap']):>8}"
            )

    # =========================================================
    # BBOX AJUSTADA + COMPARACIÓN CON BBOX CONTEXTO
    # =========================================================
    def _component_tight_bbox_from_patch_mask(self, patch_mask):
        ys, xs = np.where(patch_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        return (x1, y1, x2, y2)

    def _expand_bbox(self, bbox, img_shape, pad_x=0, pad_y=0, pad_top_extra=0):
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox

        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y - pad_top_extra)
        new_x2 = min(w, x2 + pad_x)
        new_y2 = min(h, y2 + pad_y)

        return (new_x1, new_y1, new_x2, new_y2)

    def build_adjusted_bboxes(
        self,
        pad_x_tight=8,
        pad_y_tight=6,
        top_extra_tight=10
    ):
        h, w = self.image.shape[:2]

        for comp in self.components:
            bbox_context = comp["bbox"]
            patch_mask = comp["patch_mask"]

            tight_local = self._component_tight_bbox_from_patch_mask(patch_mask)
            if tight_local is None:
                comp["bbox_tight"] = bbox_context
                continue

            px1, py1, px2, py2 = bbox_context
            tx1, ty1, tx2, ty2 = tight_local

            gx1 = px1 + tx1
            gy1 = py1 + ty1
            gx2 = px1 + tx2
            gy2 = py1 + ty2

            cy = comp["centroid_y"]
            extra_top = top_extra_tight if cy < self.top_region_ratio * h else 0

            bbox_tight = self._expand_bbox(
                (gx1, gy1, gx2, gy2),
                img_shape=self.image.shape,
                pad_x=pad_x_tight,
                pad_y=pad_y_tight,
                pad_top_extra=extra_top
            )

            comp["bbox_context"] = bbox_context
            comp["bbox_tight"] = bbox_tight

    def bbox_pair_metrics_mode(self, idx1, idx2, mode="context"):
        comp1 = self.components[idx1]
        comp2 = self.components[idx2]

        if mode == "tight":
            bbox1 = comp1.get("bbox_tight", comp1["bbox"])
            bbox2 = comp2.get("bbox_tight", comp2["bbox"])
        else:
            bbox1 = comp1.get("bbox_context", comp1["bbox"])
            bbox2 = comp2.get("bbox_context", comp2["bbox"])

        inter = self._bbox_intersection(bbox1, bbox2)
        inter_area = self._bbox_intersection_area(bbox1, bbox2)
        union_area = self._bbox_union_area(bbox1, bbox2)
        iou = self._bbox_iou(bbox1, bbox2)
        min_dist = self._bbox_min_distance(bbox1, bbox2)
        h_corners = self._bbox_hausdorff_corners(bbox1, bbox2)
        h_border = self._bbox_hausdorff_border(bbox1, bbox2)

        cx1, cy1 = comp1["centroid_x"], comp1["centroid_y"]
        cx2, cy2 = comp2["centroid_x"], comp2["centroid_y"]

        centroid_dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))
        delta_y = abs(cy1 - cy2)
        delta_x = abs(cx1 - cx2)

        return {
            "mode": mode,
            "idx1": idx1,
            "idx2": idx2,
            "bbox1": bbox1,
            "bbox2": bbox2,
            "overlap": inter is not None,
            "intersection_bbox": inter,
            "intersection_area": int(inter_area),
            "union_area": int(union_area),
            "iou": float(iou),
            "min_distance": float(min_dist),
            "centroid_distance": float(centroid_dist),
            "delta_x_centroid": float(delta_x),
            "delta_y_centroid": float(delta_y),
            "hausdorff_corners": float(h_corners),
            "hausdorff_border": float(h_border),
        }

    def compare_context_vs_tight_pairs(self, only_consecutive=True, max_rows=30):
        n = len(self.components)

        if only_consecutive:
            pairs = [(i, i + 1) for i in range(n - 1)]
        else:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        print("-" * 150)
        print(
            f"{'i':>3} {'j':>3} | "
            f"{'IoU_ctx':>8} {'IoU_tight':>10} | "
            f"{'dmin_ctx':>9} {'dmin_tight':>11} | "
            f"{'Haus_ctx':>9} {'Haus_tight':>11} | "
            f"{'ov_ctx':>7} {'ov_tight':>9}"
        )
        print("-" * 150)

        for k, (i, j) in enumerate(pairs[:max_rows]):
            m_ctx = self.bbox_pair_metrics_mode(i, j, mode="context")
            m_tight = self.bbox_pair_metrics_mode(i, j, mode="tight")

            print(
                f"{i:>3} {j:>3} | "
                f"{m_ctx['iou']:>8.4f} {m_tight['iou']:>10.4f} | "
                f"{m_ctx['min_distance']:>9.2f} {m_tight['min_distance']:>11.2f} | "
                f"{m_ctx['hausdorff_border']:>9.2f} {m_tight['hausdorff_border']:>11.2f} | "
                f"{str(m_ctx['overlap']):>7} {str(m_tight['overlap']):>9}"
            )

    def build_overlay_dual(self):
        out = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        for idx, comp in enumerate(self.components):
            bbox_context = comp.get("bbox_context", comp["bbox"])
            bbox_tight = comp.get("bbox_tight", comp["bbox"])

            x1, y1, x2, y2 = bbox_context
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 80), 2)

            tx1, ty1, tx2, ty2 = bbox_tight
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (80, 255, 255), 2)

            cx = int(round(comp["centroid_x"]))
            cy = int(round(comp["centroid_y"]))
            cv2.circle(out, (cx, cy), 3, (255, 255, 0), -1)

            cv2.putText(
                out,
                str(idx),
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        self.overlay_dual = out

    def show_overlay_dual(self, figsize=(8, 12)):
        if not hasattr(self, "overlay_dual"):
            self.build_overlay_dual()

        plt.figure(figsize=figsize)
        plt.imshow(self.overlay_dual)
        plt.title("Rojo = context | Cian = tight")
        plt.axis("off")
        plt.show()

    # =========================================================
    # MÉTRICAS POR COMPONENTE
    # =========================================================
    def component_metrics(self, idx, mode="context"):
        comp = self.components[idx]

        if mode == "tight":
            bbox = comp.get("bbox_tight", comp["bbox"])
        else:
            bbox = comp.get("bbox_context", comp["bbox"])

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        bbox_area = max(0, width) * max(0, height)

        component_area = comp["area"]
        aspect_ratio = width / height if height > 0 else 0.0
        occupancy = component_area / bbox_area if bbox_area > 0 else 0.0

        cy = comp["centroid_y"]
        cx = comp["centroid_x"]

        return {
            "idx": idx,
            "mode": mode,
            "bbox": bbox,
            "component_area": int(component_area),
            "bbox_area": int(bbox_area),
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(aspect_ratio),
            "occupancy": float(occupancy),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
        }

    def all_component_metrics(self, mode="context"):
        return [self.component_metrics(i, mode=mode) for i in range(len(self.components))]

    def print_component_metrics(self, mode="context", max_rows=50):
        rows = self.all_component_metrics(mode=mode)

        print("-" * 120)
        print(
            f"{'i':>3} | {'area':>8} | {'bbox_area':>10} | "
            f"{'w':>5} | {'h':>5} | {'aspect':>8} | {'occup':>8} | {'cy':>8}"
        )
        print("-" * 120)

        for r in rows[:max_rows]:
            print(
                f"{r['idx']:>3} | "
                f"{r['component_area']:>8} | "
                f"{r['bbox_area']:>10} | "
                f"{r['width']:>5} | "
                f"{r['height']:>5} | "
                f"{r['aspect_ratio']:>8.3f} | "
                f"{r['occupancy']:>8.3f} | "
                f"{r['centroid_y']:>8.2f}"
            )

    # =========================================================
    # CLASIFICAR COMPONENTES: "good", "doubtful", "bad"
    # =========================================================
    def classify_component_quality(
        self,
        idx,
        mode="tight",
        occupancy_min=0.18,
        occupancy_good=0.28,
        aspect_min=0.45,
        aspect_max=2.40,
        aspect_good_min=0.60,
        aspect_good_max=1.90,
        area_min=800,
        area_good_min=1400,
        y_margin_ratio=0.04,
        use_neighbor_iou=True,
        max_neighbor_iou_good=0.18,
        max_neighbor_iou_bad=0.35
    ):
        n = len(self.components)
        cm = self.component_metrics(idx, mode=mode)

        h, w = self.image.shape[:2]
        cy = cm["centroid_y"]

        component_area = cm["component_area"]
        occupancy = cm["occupancy"]
        aspect = cm["aspect_ratio"]

        reasons = []
        score = 0

        # 1) Área
        if component_area < area_min:
            reasons.append(f"area<{area_min}")
            score -= 2
        elif component_area >= area_good_min:
            score += 2
        else:
            score += 1

        # 2) Occupancy
        if occupancy < occupancy_min:
            reasons.append(f"occupancy<{occupancy_min}")
            score -= 2
        elif occupancy >= occupancy_good:
            score += 2
        else:
            score += 1

        # 3) Aspect ratio
        if aspect < aspect_min or aspect > aspect_max:
            reasons.append(f"aspect_outside_[{aspect_min},{aspect_max}]")
            score -= 2
        elif aspect_good_min <= aspect <= aspect_good_max:
            score += 2
        else:
            score += 1

        # 4) Zonas extremas superior/inferior
        y_margin = y_margin_ratio * h
        if cy < y_margin:
            reasons.append("too_high_in_image")
            score -= 1
        elif cy > (h - y_margin):
            reasons.append("too_low_in_image")
            score -= 1

        # 5) IoU con vecinos
        neighbor_ious = []

        if use_neighbor_iou and n > 1:
            if idx - 1 >= 0:
                m_prev = self.bbox_pair_metrics_mode(idx - 1, idx, mode=mode)
                neighbor_ious.append(m_prev["iou"])
            if idx + 1 < n:
                m_next = self.bbox_pair_metrics_mode(idx, idx + 1, mode=mode)
                neighbor_ious.append(m_next["iou"])

        max_neighbor_iou = max(neighbor_ious) if neighbor_ious else 0.0

        if use_neighbor_iou:
            if max_neighbor_iou > max_neighbor_iou_bad:
                reasons.append(f"neighbor_iou>{max_neighbor_iou_bad}")
                score -= 2
            elif max_neighbor_iou <= max_neighbor_iou_good:
                score += 1

        # Decisión final
        if score >= 5:
            label = "good"
        elif score >= 1:
            label = "doubtful"
        else:
            label = "bad"

        return {
            "idx": idx,
            "mode": mode,
            "label": label,
            "score": float(score),
            "reasons": reasons,
            "component_area": int(component_area),
            "occupancy": float(occupancy),
            "aspect_ratio": float(aspect),
            "centroid_y": float(cy),
            "max_neighbor_iou": float(max_neighbor_iou),
        }

    def classify_all_components(
        self,
        mode="tight",
        occupancy_min=0.18,
        occupancy_good=0.28,
        aspect_min=0.45,
        aspect_max=2.40,
        aspect_good_min=0.60,
        aspect_good_max=1.90,
        area_min=800,
        area_good_min=1400,
        y_margin_ratio=0.04,
        use_neighbor_iou=True,
        max_neighbor_iou_good=0.18,
        max_neighbor_iou_bad=0.35
    ):
        results = []
        for i in range(len(self.components)):
            r = self.classify_component_quality(
                idx=i,
                mode=mode,
                occupancy_min=occupancy_min,
                occupancy_good=occupancy_good,
                aspect_min=aspect_min,
                aspect_max=aspect_max,
                aspect_good_min=aspect_good_min,
                aspect_good_max=aspect_good_max,
                area_min=area_min,
                area_good_min=area_good_min,
                y_margin_ratio=y_margin_ratio,
                use_neighbor_iou=use_neighbor_iou,
                max_neighbor_iou_good=max_neighbor_iou_good,
                max_neighbor_iou_bad=max_neighbor_iou_bad
            )
            results.append(r)
        return results

    def print_component_quality(
        self,
        mode="tight",
        max_rows=50,
        occupancy_min=0.18,
        occupancy_good=0.28,
        aspect_min=0.45,
        aspect_max=2.40,
        aspect_good_min=0.60,
        aspect_good_max=1.90,
        area_min=800,
        area_good_min=1400,
        y_margin_ratio=0.04,
        use_neighbor_iou=True,
        max_neighbor_iou_good=0.18,
        max_neighbor_iou_bad=0.35
    ):
        rows = self.classify_all_components(
            mode=mode,
            occupancy_min=occupancy_min,
            occupancy_good=occupancy_good,
            aspect_min=aspect_min,
            aspect_max=aspect_max,
            aspect_good_min=aspect_good_min,
            aspect_good_max=aspect_good_max,
            area_min=area_min,
            area_good_min=area_good_min,
            y_margin_ratio=y_margin_ratio,
            use_neighbor_iou=use_neighbor_iou,
            max_neighbor_iou_good=max_neighbor_iou_good,
            max_neighbor_iou_bad=max_neighbor_iou_bad
        )

        print("-" * 140)
        print(
            f"{'i':>3} | {'label':>9} | {'score':>6} | "
            f"{'area':>8} | {'occup':>8} | {'aspect':>8} | "
            f"{'maxIoU':>8} | {'cy':>8} | reasons"
        )
        print("-" * 140)

        for r in rows[:max_rows]:
            print(
                f"{r['idx']:>3} | "
                f"{r['label']:>9} | "
                f"{r['score']:>6.1f} | "
                f"{r['component_area']:>8} | "
                f"{r['occupancy']:>8.3f} | "
                f"{r['aspect_ratio']:>8.3f} | "
                f"{r['max_neighbor_iou']:>8.3f} | "
                f"{r['centroid_y']:>8.2f} | "
                f"{', '.join(r['reasons']) if r['reasons'] else '-'}"
            )

    def build_quality_overlay(
        self,
        mode="tight",
        occupancy_min=0.18,
        occupancy_good=0.28,
        aspect_min=0.45,
        aspect_max=2.40,
        aspect_good_min=0.60,
        aspect_good_max=1.90,
        area_min=800,
        area_good_min=1400,
        y_margin_ratio=0.04,
        use_neighbor_iou=True,
        max_neighbor_iou_good=0.18,
        max_neighbor_iou_bad=0.35
    ):
        quality_rows = self.classify_all_components(
            mode=mode,
            occupancy_min=occupancy_min,
            occupancy_good=occupancy_good,
            aspect_min=aspect_min,
            aspect_max=aspect_max,
            aspect_good_min=aspect_good_min,
            aspect_good_max=aspect_good_max,
            area_min=area_min,
            area_good_min=area_good_min,
            y_margin_ratio=y_margin_ratio,
            use_neighbor_iou=use_neighbor_iou,
            max_neighbor_iou_good=max_neighbor_iou_good,
            max_neighbor_iou_bad=max_neighbor_iou_bad
        )

        qmap = {r["idx"]: r for r in quality_rows}

        out = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        for idx, comp in enumerate(self.components):
            if mode == "tight":
                bbox = comp.get("bbox_tight", comp["bbox"])
            else:
                bbox = comp.get("bbox_context", comp["bbox"])

            x1, y1, x2, y2 = bbox
            q = qmap[idx]["label"]

            if q == "good":
                color = (80, 255, 80)
            elif q == "doubtful":
                color = (255, 255, 80)
            else:
                color = (255, 80, 80)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            cx = int(round(comp["centroid_x"]))
            cy = int(round(comp["centroid_y"]))
            cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1)

            txt = f"{idx}:{q[0].upper()}"
            cv2.putText(
                out,
                txt,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

        self.quality_overlay = out

    def show_quality_overlay(self, figsize=(8, 12)):
        if not hasattr(self, "quality_overlay"):
            raise ValueError("Primero llama build_quality_overlay().")

        plt.figure(figsize=figsize)
        plt.imshow(self.quality_overlay)
        plt.title("Verde=good | Amarillo=doubtful | Rojo=bad")
        plt.axis("off")
        plt.show()

    def get_good_component_indices(
        self,
        mode="tight",
        occupancy_min=0.18,
        occupancy_good=0.28,
        aspect_min=0.45,
        aspect_max=2.40,
        aspect_good_min=0.60,
        aspect_good_max=1.90,
        area_min=800,
        area_good_min=1400,
        y_margin_ratio=0.04,
        use_neighbor_iou=True,
        max_neighbor_iou_good=0.18,
        max_neighbor_iou_bad=0.35
    ):
        rows = self.classify_all_components(
            mode=mode,
            occupancy_min=occupancy_min,
            occupancy_good=occupancy_good,
            aspect_min=aspect_min,
            aspect_max=aspect_max,
            aspect_good_min=aspect_good_min,
            aspect_good_max=aspect_good_max,
            area_min=area_min,
            area_good_min=area_good_min,
            y_margin_ratio=y_margin_ratio,
            use_neighbor_iou=use_neighbor_iou,
            max_neighbor_iou_good=max_neighbor_iou_good,
            max_neighbor_iou_bad=max_neighbor_iou_bad
        )
        return [r["idx"] for r in rows if r["label"] == "good"]
