import papermill as pm
import os

target_dir = "./project-2/aibi-dhi-simulator"
notebooks = [f for f in os.listdir(target_dir) if f.endswith(".ipynb")]

for nb in notebooks:
    file_path = os.path.join(target_dir, nb)
    print(f"Executing and overwriting: {nb}...")

    try:
        pm.execute_notebook(
            input_path=file_path,
            output_path=file_path,
        )
        print(f"Done: {nb}")
    except Exception as e:
        print(f"Failed to run {nb}. Error: {e}")

print("\nAll notebooks have been updated.")
