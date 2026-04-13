import os
from pathlib import Path

class PatchesProcessor:
    def __init__(self, base_dir, patient_id):
        self.base_dir = Path(base_dir)
        self.patient_id = patient_id
        self.output_dir = self.base_dir / f"patches_processor_{self.patient_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, data):
        # Aquí iría la lógica de procesamiento de parches
        # Por ejemplo, guardar un archivo de ejemplo
        output_file = self.output_dir / f"result_{self.patient_id}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Processed data for patient {self.patient_id}\n")
        return output_file
