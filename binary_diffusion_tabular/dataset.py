from typing import List, Tuple, Union, Optional

import pandas as pd

import torch
from torch.utils.data import Dataset

from binary_diffusion_tabular.transformation import FixedSizeBinaryTableTransformation, TASK


class FixedSizeBinaryTableDataset(Dataset):

    """Pytorch dataset for fixed size binary tables."""

    def __init__(
        self,
        *,
        table: pd.DataFrame,
        target_column: Optional[str] = None,
        split_feature_target: bool,
        task: TASK,
        numerical_columns: List[str] = None,
        categorical_columns: List[str] = None,
    ):
        """
        Args:
            table: pandas dataframe with categorical and numerical columns. Dataframe should not have nan
            target_column: name of the target column. Optional. Should be provided if split_feature_target is True.
            split_feature_target: split features columns and target column
            task: task for which dataset is used. Can be 'classification' or 'regression'
            numerical_columns: list of columns with numerical values
            categorical_columns: list of columns with categorical values
        """

        if numerical_columns is None:
            numerical_columns = []

        if categorical_columns is None:
            categorical_columns = []

        self.table = table
        self.target_column = target_column
        self.split_feature_target = split_feature_target
        self.task = task
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        self.transformation = FixedSizeBinaryTableTransformation(task, numerical_columns, categorical_columns)

        if self.split_feature_target:
            target = self.table[self.target_column]
            features = self.table.drop(columns=[self.target_column])

            self.features_binary, self.targets_binary = self.transformation.fit_transform(features, target)
        else:
            self.features_binary = self.transformation.fit_transform(self.table)

    @property
    def n_classes(self) -> int:
        return self.transformation.n_classes

    @property
    def row_size(self) -> int:
        return self.transformation.row_size

    @property
    def conditional(self) -> bool:
        return self.split_feature_target

    def __len__(self) -> int:
        return len(self.features_binary)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        row = self.features_binary[idx]

        if self.split_feature_target:
            target = self.targets_binary[idx]
            return row, target
        return row
