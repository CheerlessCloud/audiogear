import os

import pytest

from audiogear.executer.local import LocalPipelineExecutor


@pytest.mark.parametrize(
    "visible_devices,local_rank,expected",
    [
        ("7", 0, "7"),
        ("2,3", 0, "2"),
        ("2,3", 1, "3"),
        ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,GPU-11111111-2222-3333-4444-555555555555", 1,
         "GPU-11111111-2222-3333-4444-555555555555"),
    ],
)
def test_gpu_pinning_honors_inherited_visible_device_tokens(
    tmp_path,
    monkeypatch,
    visible_devices,
    local_rank,
    expected,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    executor = LocalPipelineExecutor(
        pipeline=[],
        logging_dir=str(tmp_path),
        gpus=2,
        workers=2,
    )

    executor._pin_gpu(local_rank)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_gpu_pinning_uses_physical_index_only_without_an_inherited_mask(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    executor = LocalPipelineExecutor(
        pipeline=[],
        logging_dir=str(tmp_path),
        gpus=2,
        workers=2,
    )

    executor._pin_gpu(1)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
