import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd

from everyai.data_loader.data_load import Data_loader


class TestDataLoader(unittest.TestCase):
    @patch("everyai.data_loader.data_load.pd.read_csv")
    def test_load_csv(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({"question": ["Q1"], "answer": ["A1"], "model": ["M1"]})
        loader = Data_loader("test.csv", "question", "answer", "model")
        data = loader.load_data2list()
        mock_read_csv.assert_called_once_with("test.csv")
        self.assertIsInstance(data, pd.DataFrame)

    @patch("everyai.data_loader.data_load.pd.read_excel")
    def test_load_excel(self, mock_read_excel):
        mock_read_excel.return_value = pd.DataFrame({"question": ["Q1"], "answer": ["A1"], "model": ["M1"]})
        loader = Data_loader("test.xlsx", "question", "answer", "model")
        data = loader.load_data2list()
        mock_read_excel.assert_called_once_with("test.xlsx")
        self.assertIsInstance(data, pd.DataFrame)

    @patch("everyai.data_loader.data_load.pd.read_json")
    def test_load_jsonl(self, mock_read_json):
        mock_read_json.return_value = pd.DataFrame({"question": ["Q1"], "answer": ["A1"], "model": ["M1"]})
        loader = Data_loader("test.jsonl", "question", "answer", "model")
        data = loader.load_data2list()
        mock_read_json.assert_called_once_with("test.jsonl", lines=True)
        self.assertIsInstance(data, pd.DataFrame)

    @patch("everyai.data_loader.data_load.pd.read_json")
    def test_load_json(self, mock_read_json):
        mock_read_json.return_value = pd.DataFrame({"question": ["Q1"], "answer": ["A1"], "model": ["M1"]})
        loader = Data_loader("test.json", "question", "answer", "model")
        data = loader.load_data2list()
        mock_read_json.assert_called_once_with("test.json")
        self.assertIsInstance(data, pd.DataFrame)

    @patch("everyai.data_loader.data_load.load_dataset")
    def test_load_dataset(self, mock_load_dataset):
        mock_load_dataset.return_value = {"train": [{"question": "Q1", "answer": "A1", "model": "M1"}]}
        loader = Data_loader("dataset_name", "question", "answer", "model")
        data = loader.load_data2list()
        mock_load_dataset.assert_called_once_with("dataset_name")
        self.assertIsInstance(data, list)

    @patch("everyai.data_loader.data_load.Path.exists")
    def test_invalid_file_format(self, mock_path_exists):
        mock_path_exists.return_value = True
        loader = Data_loader("test.invalid", "question", "answer", "model")
        with self.assertLogs(level='ERROR') as log:
            data = loader.load_data2list()
            self.assertIn("Invalid file format", log.output[0])
        self.assertIsNone(data)

if __name__ == "__main__":
    unittest.main()