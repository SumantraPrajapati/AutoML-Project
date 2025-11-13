from cli.cli_interface import app
import multiprocessing


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app()
