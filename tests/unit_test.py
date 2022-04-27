import unittest

import yaml


def read(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data_path(config_path):
    config = read(config_path)
    # print(config)
    data_path = config["load_and_split_data"]["deps"]
    return data_path


class MyTest(unittest.TestCase):
    def test(self):
        housing_path = get_data_path("env.yaml")
        self.assertEqual(housing_path, "data")
