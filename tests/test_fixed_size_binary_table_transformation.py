import unittest
import os
import tempfile

import pandas as pd
import torch
import numpy as np

from binary_diffusion_tabular import FixedSizeBinaryTableTransformation


def create_sample_data():
    """Create a sample DataFrame for testing based on adult dataset."""
    data = {
        "age": [25, 38, 28, 44, 18],
        "fnlwgt": [226802, 89814, 336951, 160323, 103497],
        "education-num": [7, 9, 10, 10, 9],
        "capital-gain": [0, 0, 0, 7688, 0],
        "capital-loss": [1, 2, 3, 4, 5],
        "hours-per-week": [40, 50, 40, 40, 30],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "Private", "Private"],
        "education": ["Bachelors", "HS-grad", "11th", "Masters", "HS-grad"],
        "marital-status": [
            "Never-married",
            "Married-civ-spouse",
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
        ],
        "occupation": [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Adm-clerical",
        ],
        "relationship": [
            "Not-in-family",
            "Husband",
            "Husband",
            "Unmarried",
            "Not-in-family",
        ],
        "race": ["White", "White", "Asian-Pac-Islander", "Black", "White"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "native-country": [
            "United-States",
            "United-States",
            "United-States",
            "United-States",
            "United-States",
        ],
        "label": [1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    return df


class TestFixedSizeBinaryTableTransformation(unittest.TestCase):
    def setUp(self):
        """Set up sample data and transformation instance before each test."""
        self.df = create_sample_data()
        self.numerical_cols = [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        self.categorical_cols = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        self.transformation = FixedSizeBinaryTableTransformation(
            task="classification",
            numerical_columns=self.numerical_cols,
            categorical_columns=self.categorical_cols,
            parallel=False,  # Change to True to test parallel execution
        )

    def test_fit_transform_and_transform_consistency(self):
        """Test that fit_transform and transform methods produce consistent results."""
        df_y = self.df["label"]
        df_x = self.df.drop("label", axis=1)

        x_binary, y_trans = self.transformation.fit_transform(df_x, df_y)
        x_binary_2, y_trans_2 = self.transformation.transform(df_x, df_y)
        self.assertTrue(
            torch.all(x_binary == x_binary_2), "x_binary and x_binary_2 should be equal"
        )
        self.assertTrue(
            torch.all(y_trans == y_trans_2), "y_trans and y_trans_2 should be equal"
        )

    def test_inverse_transform(self):
        """Test that inverse_transform accurately retrieves the original data."""
        df_y = self.df["label"]
        df_x = self.df.drop("label", axis=1)

        x_binary, y_trans = self.transformation.fit_transform(df_x, df_y)
        df_x_back, y_back = self.transformation.inverse_transform(x_binary, y_trans)

        for col in self.categorical_cols:
            original = self.df[col].reset_index(drop=True)
            back = df_x_back[col].reset_index(drop=True)
            self.assertTrue(
                original.equals(back),
                f"Categorical column '{col}' does not match after inverse transform",
            )

        for col in self.numerical_cols:
            original = self.df[col].values
            back = df_x_back[col].values
            self.assertTrue(
                np.allclose(original, back, atol=1e-5),
                f"Numerical column '{col}' does not match after inverse transform",
            )

    def test_parallel_transformation(self):
        """Test that parallel and non-parallel transformations produce the same results."""
        df_y = self.df["label"]
        df_x = self.df.drop("label", axis=1)

        self.transformation.parallel = False
        x_binary, y_trans = self.transformation.fit_transform(df_x, df_y)

        transformation_parallel = FixedSizeBinaryTableTransformation(
            task="classification",
            numerical_columns=self.numerical_cols,
            categorical_columns=self.categorical_cols,
            parallel=True,
        )
        x_binary_p, y_trans_p = transformation_parallel.fit_transform(df_x, df_y)

        self.assertTrue(
            torch.all(x_binary == x_binary_p),
            "Binary tensors should be equal when using parallel and non-parallel transforms",
        )
        self.assertTrue(
            torch.all(y_trans == y_trans_p),
            "Labels should be equal when using parallel and non-parallel transforms",
        )

    def test_invalid_numerical_dtype(self):
        """Test that a ValueError is raised when numerical columns have non-numeric types."""
        df_x = self.df.drop("label", axis=1).copy()
        df_x["age"] = df_x["age"].astype(str)  # Introduce invalid dtype

        with self.assertRaises(ValueError):
            self.transformation.fit_transform(df_x, self.df["label"])

    def test_transform_without_fit(self):
        """Test that transforming without fitting raises a RuntimeError."""
        df_x = self.df.drop("label", axis=1)

        with self.assertRaises(RuntimeError):
            self.transformation.transform(df_x)

    def test_inverse_transform_without_fit(self):
        """Test that inverse_transform without fitting raises a RuntimeError."""
        transformation_unfitted = FixedSizeBinaryTableTransformation(
            task="classification",
            numerical_columns=self.numerical_cols,
            categorical_columns=self.categorical_cols,
            parallel=False,
        )
        fake_tensor = torch.zeros((5, 32))

        with self.assertRaises(RuntimeError):
            transformation_unfitted.inverse_transform(fake_tensor)

    def test_label_transformation(self):
        """Test that labels are correctly transformed and inverse transformed."""
        df_y = self.df["label"]
        df_x = self.df.drop("label", axis=1)

        x_binary, y_trans = self.transformation.fit_transform(df_x, df_y)
        y_back = self.transformation.inverse_transform_label(y_trans)

        original_labels = df_y.values
        self.assertTrue(
            np.array_equal(original_labels, y_back),
            "Original labels and inverse transformed labels should match",
        )

    def test_save_and_load_transformation(self):
        """Test that saving and loading the transformation preserves its state and functionality."""
        df_y = self.df["label"]
        df_x = self.df.drop("label", axis=1)

        x_binary_original, y_trans_original = self.transformation.fit_transform(
            df_x, df_y
        )

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            temp_filepath = tmp_file.name

        try:
            self.transformation.save_checkpoint(temp_filepath)

            loaded_transformation = FixedSizeBinaryTableTransformation.from_checkpoint(
                temp_filepath
            )

            self.assertTrue(
                loaded_transformation.fitted, "Loaded transformer should be fitted."
            )
            self.assertTrue(
                loaded_transformation.fitted_label,
                "Loaded transformer should have fitted labels.",
            )

            x_binary_loaded, y_trans_loaded = loaded_transformation.transform(
                df_x, df_y
            )

            self.assertTrue(
                torch.all(x_binary_original == x_binary_loaded),
                "Transformed data from loaded transformer should match the original transformer.",
            )
            self.assertTrue(
                torch.all(y_trans_original == y_trans_loaded),
                "Transformed labels from loaded transformer should match the original transformer.",
            )

            df_x_back_loaded, y_back_loaded = loaded_transformation.inverse_transform(
                x_binary_loaded, y_trans_loaded
            )

            for col in self.categorical_cols:
                original = self.df[col].reset_index(drop=True)
                back = df_x_back_loaded[col].reset_index(drop=True)
                self.assertTrue(
                    original.equals(back),
                    f"Categorical column '{col}' does not match after inverse transform with loaded transformer",
                )

            for col in self.numerical_cols:
                original = self.df[col].values
                back = df_x_back_loaded[col].values
                self.assertTrue(
                    np.allclose(original, back, atol=1e-5),
                    f"Numerical column '{col}' does not match after inverse transform with loaded transformer",
                )

        finally:
            # Clean up the temporary file
            os.remove(temp_filepath)


if __name__ == "__main__":
    unittest.main()
