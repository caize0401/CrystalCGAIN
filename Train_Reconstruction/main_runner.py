import subprocess


def run_program(file_name):
    try:
        # Use subprocess.run to execute the external program
        result = subprocess.run(["python", file_name], check=True)
        print(f"{file_name} finished running, return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {file_name}: {e}")
    except FileNotFoundError:
        print(f"File {file_name} not found. Please check the path.")


if __name__ == "__main__":
    print("Running program 1...")
    run_program("crystal_main.py")

    print("Running program 2...")
    run_program("plt_show_train.py")

