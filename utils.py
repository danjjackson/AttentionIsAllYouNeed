import argparse
import os
import importlib.util
import sys

def parse_args(cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        type=str,
        metavar='config.py',
        help='Python configuration file for the experiment'
    )
    parser.add_argument(
        '-n',
        '--num_epochs', 
        type=int, 
        default=20,
        help='Number of epochs to train the model for'
        )
    return parser.parse_args(cmdline)

def load_config(filename):
    if not os.path.exists(filename):
        raise ValueError(f"The config sepcified does not exist: {filename}")
    if not os.path.isfile(filename):
        raise ValueError(f"The config specified is not a file: {filename}")
    if os.path.splitext(filename)[1] != ".py":
        raise ValueError("The config specified is not a python file")

    module_name = os.path.splitext(os.path.basename(filename))[0]
    if module_name in sys.modules:
        print(f"A module with the same name as '{module_name}' already exists")
        raise ImportError(f"Cannot import {module_name} as it already exists")

    config_dir = os.path.dirname(filename)
    if config_dir not in sys.path:
        sys.path.append(config_dir)
    spec = importlib.util.spec_from_file_location(module_name, filename)
    sys.modules[module_name] = importlib.util.module_from_spec(spec)
    config_module = spec.loader.load_module(module_name)
    pyconf = getattr(config_module, 'config')

    return pyconf