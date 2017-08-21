import ruamel.yaml as yaml
import codecs
import os

def parse_config(config_path, loader=yaml.Loader):
    if not os.path.exists(config_path):
        raise ValueError("config file not found in {config_path}".format(config_path))
    with codecs.open(config_path, encoding="utf-8") as stream:
        config = yaml.load(stream, loader)
    return config
