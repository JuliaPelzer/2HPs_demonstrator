import pytest
import torch
from pipeline import *


def test_norm():
    (
        datasets_raw_domain_dir,
        datasets_prepared_domain_dir,
        dataset_domain_path,
        datasets_model_trained_with_path,
        model_path,
        name_extension,
        datasets_prepared_2hp_dir,
    ) = set_paths(
        "dataset_2hps_1fixed_testing",
        "current_unet_benchmark_dataset_2d_100datapoints_input_empty_T_0",
        "benchmark_dataset_2d_100datapoints_input_empty_T_0",
        "g",
    )
    dummy_domain = Domain(dataset_domain_path, stitching_method="max")

    test_value = [1, 3.5, 0.27]
    test_value = torch.tensor(test_value)
    result = dummy_domain.reverse_norm(dummy_domain.norm(test_value))

    assert torch.allclose(result, test_value)
