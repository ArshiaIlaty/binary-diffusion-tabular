import unittest

import torch
import torch.nn.functional as F

from binary_diffusion_tabular import SimpleTableGenerator


class TestSimpleTableGenerator(unittest.TestCase):
    def test_classification_conditional_without_classifier_free_guidance(self):
        """
        Test SimpleTableGenerator for classification task with conditional=True and classifier_free_guidance=False.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="classification",
            conditional=True,
            n_classes=3,
            classifier_free_guidance=False,
        )
        batch_size = 128
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()
        cls = torch.randint(0, 3, (batch_size,))

        out = model(tensor, ts, cls)
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )

    def test_classification_conditional_with_classifier_free_guidance(self):
        """
        Test SimpleTableGenerator for classification task with conditional=True and classifier_free_guidance=True.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="classification",
            conditional=True,
            n_classes=3,
            classifier_free_guidance=True,
        )
        batch_size = 128
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()
        cls = torch.randint(0, 3, (batch_size,))

        # for classifier free guidance, cls should be one-hot
        cls = F.one_hot(cls, num_classes=3).float()

        out = model(tensor, ts, cls)
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )

    def test_classification_unconditional(self):
        """
        Test SimpleTableGenerator for classification task with conditional=False.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="classification",
            conditional=False,
            n_classes=0,  # Irrelevant since conditional=False
            classifier_free_guidance=False,  # Irrelevant since conditional=False
        )
        batch_size = 128
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()

        out = model(tensor, ts)  # No class labels provided
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )

    def test_regression_conditional(self):
        """
        Test SimpleTableGenerator for regression task with conditional=True.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="regression",
            conditional=True,
            n_classes=0,  # Irrelevant for regression
            classifier_free_guidance=False,  # Irrelevant for regression
        )
        batch_size = 128
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()
        reg = torch.randn((batch_size, 1))  # Regression targets

        out = model(tensor, ts, reg)
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )

    def test_regression_unconditional(self):
        """
        Test SimpleTableGenerator for regression task with conditional=False.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="regression",
            conditional=False,
            n_classes=0,  # Irrelevant since conditional=False
            classifier_free_guidance=False,  # Irrelevant since conditional=False
        )
        batch_size = 128
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()

        out = model(tensor, ts)  # No regression targets provided
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )

    def test_invalid_task(self):
        """
        Test that providing an invalid task raises an error.
        """
        with self.assertRaises(ValueError):
            SimpleTableGenerator(
                data_dim=220,
                dim=256,
                n_res_blocks=3,
                out_dim=220,
                task="invalid_task",  # Invalid task
                conditional=True,
                n_classes=3,
                classifier_free_guidance=False,
            )

    def test_incorrect_class_count(self):
        """
        Test that providing a non-positive number of classes raises an error for classification.
        """
        with self.assertRaises(ValueError):
            SimpleTableGenerator(
                data_dim=220,
                dim=256,
                n_res_blocks=3,
                out_dim=220,
                task="classification",
                conditional=True,
                n_classes=0,  # Invalid number of classes
                classifier_free_guidance=False,
            )

    def test_output_dtype(self):
        """
        Test that the output tensor has the correct dtype.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="classification",
            conditional=True,
            n_classes=3,
            classifier_free_guidance=False,
        )
        batch_size = 128
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()
        cls = torch.randint(0, 3, (batch_size,))

        out = model(tensor, ts, cls)
        expected_dtype = torch.float32  # Assuming the model outputs float tensors
        self.assertEqual(
            out.dtype,
            expected_dtype,
            f"Output dtype should be {expected_dtype}, got {out.dtype}",
        )

    def test_batch_size_zero(self):
        """
        Test that the model can handle a batch size of zero without errors.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="classification",
            conditional=True,
            n_classes=3,
            classifier_free_guidance=False,
        )
        batch_size = 0
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()
        cls = torch.randint(0, 3, (batch_size,))

        out = model(tensor, ts, cls)
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )

    def test_large_batch_size(self):
        """
        Test that the model can handle a large batch size without errors.
        """
        model = SimpleTableGenerator(
            data_dim=220,
            dim=256,
            n_res_blocks=3,
            out_dim=220,
            task="classification",
            conditional=True,
            n_classes=3,
            classifier_free_guidance=False,
        )
        batch_size = 1024
        tensor = torch.randn((batch_size, 220))
        ts = torch.randint(0, 100, (batch_size,)).float()
        cls = torch.randint(0, 3, (batch_size,))

        out = model(tensor, ts, cls)
        expected_shape = (batch_size, 220)
        self.assertEqual(
            out.shape,
            expected_shape,
            f"Output shape should be {expected_shape}, got {out.shape}",
        )


if __name__ == "__main__":
    unittest.main()
