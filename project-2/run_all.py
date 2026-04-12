import papermill as pm
import os

target_dir = "./aibi-dhi-simulator"
# Get the absolute path to ensure no ambiguity
abs_target_dir = os.path.abspath(target_dir)

notebooks = [f for f in os.listdir(abs_target_dir) if f.endswith(".ipynb")]

for nb in notebooks:
    file_path = os.path.join(abs_target_dir, nb)
    print(f"Executing: {nb}...")

    try:
        pm.execute_notebook(
            input_path=file_path,
            output_path=file_path,
            cwd=abs_target_dir,
        )
        print(f"Done: {nb}")
    except Exception as e:
        print(f"Failed: {nb}. Error: {e}")
