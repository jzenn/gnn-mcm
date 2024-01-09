import argparse
import json


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        f = json.load(f)
    return f


def dump_json(file: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(file, f, indent=4)


def save_arguments_to_path(args: argparse.Namespace, path: str) -> None:
    # adapted from https://stackoverflow.com/a/55114771
    dump_json(args.__dict__, path)
