

###################################### fsl_reorient_only_final.py

import os
import glob

temporary_folder = "./convert_nifti_format"
bash_commands = []

# Delete old reoriented files first
print("Cleaning old reoriented files...")
for patient_id in os.listdir(temporary_folder):
    patient_dir = os.path.join(temporary_folder, patient_id)
    if os.path.isdir(patient_dir):
        for f in os.listdir(patient_dir):
            if "_re" in f:
                os.remove(os.path.join(patient_dir, f))

patient_folders = sorted(os.listdir(temporary_folder))

for patient_id in patient_folders:
    patient_nifti_dir = os.path.join(temporary_folder, patient_id)
    
    if not os.path.isdir(patient_nifti_dir):
        continue
    
    mask_files = glob.glob(os.path.join(patient_nifti_dir, "mask_*.nii.gz"))
    mask_files = [f for f in mask_files if "_re" not in f]
    
    if not mask_files:
        continue
    
    mask_file = os.path.basename(mask_files[0])
    structure_name = mask_file.replace("mask_", "").replace(".nii.gz", "")
    
    # ONLY reorient - save back to same folder
    bash_commands.append(
        f"fslreorient2std "
        f"/data/convert_nifti_format/{patient_id}/image.nii.gz "
        f"/data/convert_nifti_format/{patient_id}/image_re.nii.gz"
    )
    
    bash_commands.append(
        f"fslreorient2std "
        f"/data/convert_nifti_format/{patient_id}/{mask_file} "
        f"/data/convert_nifti_format/{patient_id}/mask_{structure_name}_re.nii.gz"
    )

with open("fsl_reorient_final.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for cmd in bash_commands:
        f.write(cmd + "\n")

print(f"✅ Generated {len(bash_commands)} commands (reorient only)")




# ******************* AFTER RUNNING THE CODE PLEASE RUN THIS LINE INTO "VS Code" TERMINAL ********************************

  # run python fsl_docker.py
  # sudo docker run -v $(pwd):/data -w /data brainlife/fsl bash /data/fsl_reorient_final.sh``