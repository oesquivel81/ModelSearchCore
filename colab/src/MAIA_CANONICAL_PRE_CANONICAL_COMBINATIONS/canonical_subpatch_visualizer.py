class CanonicalSubpatchPipeline:
    def __init__(self, metadata_csv, save_root, patch_size=(128,128), subpatch_size=(32,32), stride=(32,32)):
        self.metadata_csv = metadata_csv
        self.save_root = save_root
        self.patch_size = patch_size
        self.subpatch_size = subpatch_size
        self.stride = stride
        os.makedirs(save_root, exist_ok=True)

    def run(self):
        from colab.src.extractor.vertebra_subpatch_generator import VertebraSubpatchGenerator

        df = pd.read_csv(self.metadata_csv)
        rows_out = []

        for _, row in df.iterrows():
            img_path = row.get("filtered_patch_path", row.get("image_patch_path", None))
            if not isinstance(img_path, str) or not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            gen = VertebraSubpatchGenerator(
                patch_size=self.patch_size,
                subpatch_size=self.subpatch_size,
                stride=self.stride
            )

            img_patch = gen._resize(img)
            windows = gen._extract_windows(img_patch)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = os.path.join(self.save_root, base_name)
            os.makedirs(out_dir, exist_ok=True)

            for i, (r, c, x1, y1, x2, y2, sub) in enumerate(windows):
                sub_path = os.path.join(out_dir, f"subpatch_{i:03d}.png")
                cv2.imwrite(sub_path, sub)

                rows_out.append({
                    "source_patch_path": img_path,
                    "subpatch_index": i,
                    "grid_r": r,
                    "grid_c": c,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "subpatch_path": sub_path
                })

        if rows_out:
            pd.DataFrame(rows_out).to_csv(
                os.path.join(self.save_root, "all_subpatches_metadata.csv"),
                index=False
            )