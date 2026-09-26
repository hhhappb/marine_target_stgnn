from __future__ import annotations

import pytest

from paper_modules.datasets.registry import build_dataset, reject_retired_ipix_cross_file_split


@pytest.mark.parametrize(
    "retired_key",
    [
        "train_sources",
        "test_sources",
        "require_disjoint_train_test_sources",
    ],
)
def test_retired_ipix_cross_file_split_fields_fail_loud(retired_key: str) -> None:
    with pytest.raises(ValueError, match="跨文件训练/测试协议已退役"):
        reject_retired_ipix_cross_file_split({retired_key: ["19931107"]})


def test_shared_ipix_sources_remain_supported() -> None:
    reject_retired_ipix_cross_file_split({"sources": ["19931107"]})


def test_dataset_entrypoint_rejects_retired_cross_file_split() -> None:
    config = {"dataset": {"type": "ipix_window", "train_sources": ["19931107"]}}
    with pytest.raises(ValueError, match="跨文件训练/测试协议已退役"):
        build_dataset(config, "train")
