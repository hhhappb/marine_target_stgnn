from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import get_config_value, load_config


NC_DIMENSION = 10
NC_VARIABLE = 11
NC_ATTRIBUTE = 12

NC_BYTE = 1
NC_CHAR = 2
NC_SHORT = 3
NC_INT = 4
NC_FLOAT = 5
NC_DOUBLE = 6

TYPE_SIZES = {
    NC_BYTE: 1,
    NC_CHAR: 1,
    NC_SHORT: 2,
    NC_INT: 4,
    NC_FLOAT: 4,
    NC_DOUBLE: 8,
}

TYPE_NAMES = {
    NC_BYTE: "byte",
    NC_CHAR: "char",
    NC_SHORT: "short",
    NC_INT: "int",
    NC_FLOAT: "float",
    NC_DOUBLE: "double",
}

POLARIZATIONS = ("hh", "hv", "vv", "vh")


@dataclass(frozen=True)
class NetCDFVariable:
    name: str
    dim_ids: tuple[int, ...]
    attrs: dict[str, Any]
    nc_type: int
    size: int
    begin: int


class HeaderReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_u32(self) -> int:
        value = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4
        return value

    def read_name(self) -> str:
        length = self.read_u32()
        raw = self.data[self.offset : self.offset + length]
        self.offset += length
        self.offset += (-length) % 4
        return raw.decode("ascii")

    def read_values(self, nc_type: int, count: int) -> Any:
        size = TYPE_SIZES[nc_type] * count
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        self.offset += (-size) % 4

        if nc_type == NC_CHAR:
            return raw.decode("ascii").rstrip("\x00")
        if nc_type == NC_BYTE:
            values = struct.unpack(f">{count}b", raw)
        elif nc_type == NC_SHORT:
            values = struct.unpack(f">{count}h", raw)
        elif nc_type == NC_INT:
            values = struct.unpack(f">{count}i", raw)
        elif nc_type == NC_FLOAT:
            values = struct.unpack(f">{count}f", raw)
        elif nc_type == NC_DOUBLE:
            values = struct.unpack(f">{count}d", raw)
        else:
            raise ValueError(f"Unsupported NetCDF type id {nc_type}")
        return values[0] if count == 1 else values

    def read_attrs(self) -> dict[str, Any]:
        tag = self.read_u32()
        count = self.read_u32()
        if tag == 0 and count == 0:
            return {}
        if tag != NC_ATTRIBUTE:
            raise ValueError(f"Expected NetCDF attribute tag, got {tag}")

        attrs: dict[str, Any] = {}
        for _ in range(count):
            name = self.read_name()
            nc_type = self.read_u32()
            value_count = self.read_u32()
            attrs[name] = self.read_values(nc_type, value_count)
        return attrs


class NetCDFClassicFile:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.version = 0
        self.num_records = 0
        self.dims: list[tuple[str, int]] = []
        self.attrs: dict[str, Any] = {}
        self.variables: dict[str, NetCDFVariable] = {}
        self._parse_header()

    def _parse_header(self) -> None:
        if self.data[:3] != b"CDF" or self.data[3] not in (1, 2):
            raise ValueError(f"{self.path} is not a NetCDF classic CDF-1/CDF-2 file")

        self.version = self.data[3]
        reader = HeaderReader(self.data)
        reader.offset = 4
        self.num_records = reader.read_u32()

        dim_tag = reader.read_u32()
        dim_count = reader.read_u32()
        if dim_tag != NC_DIMENSION:
            raise ValueError(f"{self.path} has no dimension list")
        for _ in range(dim_count):
            self.dims.append((reader.read_name(), reader.read_u32()))

        self.attrs = reader.read_attrs()

        var_tag = reader.read_u32()
        var_count = reader.read_u32()
        if var_tag != NC_VARIABLE:
            raise ValueError(f"{self.path} has no variable list")

        for _ in range(var_count):
            name = reader.read_name()
            dim_count = reader.read_u32()
            dim_ids = tuple(reader.read_u32() for _ in range(dim_count))
            attrs = reader.read_attrs()
            nc_type = reader.read_u32()
            size = reader.read_u32()
            begin = reader.read_u32() if self.version == 1 else self._read_u64(reader)
            self.variables[name] = NetCDFVariable(name, dim_ids, attrs, nc_type, size, begin)

    @staticmethod
    def _read_u64(reader: HeaderReader) -> int:
        value = struct.unpack(">Q", reader.data[reader.offset : reader.offset + 8])[0]
        reader.offset += 8
        return value

    def shape_of(self, variable: NetCDFVariable) -> tuple[int, ...]:
        return tuple(self.dims[dim_id][1] for dim_id in variable.dim_ids)

    def read_variable(self, name: str) -> Any:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: numpy. Install with `python -m pip install -r requirements.txt`."
            ) from exc

        variable = self.variables[name]
        shape = self.shape_of(variable)
        count = math.prod(shape) if shape else 1
        raw = memoryview(self.data)[variable.begin : variable.begin + TYPE_SIZES[variable.nc_type] * count]

        if variable.nc_type == NC_BYTE:
            dtype = np.uint8
        elif variable.nc_type == NC_SHORT:
            dtype = ">i2"
        elif variable.nc_type == NC_INT:
            dtype = ">i4"
        elif variable.nc_type == NC_FLOAT:
            dtype = ">f4"
        elif variable.nc_type == NC_DOUBLE:
            dtype = ">f8"
        elif variable.nc_type == NC_CHAR:
            return bytes(raw).decode("ascii").rstrip("\x00")
        else:
            raise ValueError(f"Unsupported NetCDF type {variable.nc_type}")

        values = np.frombuffer(raw, dtype=dtype, count=count)
        values = values.astype(values.dtype.newbyteorder("="), copy=False)
        return values.reshape(shape) if shape else values[0]


def load_labels(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["files"]


def make_range_labels(entry: dict[str, Any], nrange: int, target_policy: str) -> tuple[Any, Any]:
    import numpy as np

    labels = np.zeros(nrange, dtype=np.uint8)
    roles = np.zeros(nrange, dtype=np.int8)

    secondary = [int(item) for item in entry["secondary"]]
    primary = int(entry["primary"])

    for one_based in secondary:
        labels[one_based - 1] = 1
        roles[one_based - 1] = 1

    roles[primary - 1] = 2
    if target_policy == "primary":
        labels[:] = 0
        labels[primary - 1] = 1

    return labels, roles


def auto_process_iq(i_raw: Any, q_raw: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    i_values = i_raw.astype(np.float32)
    q_values = q_raw.astype(np.float32)

    mean_i = i_values.mean(axis=0, dtype=np.float64)
    mean_q = q_values.mean(axis=0, dtype=np.float64)
    std_i = i_values.std(axis=0, dtype=np.float64)
    std_q = q_values.std(axis=0, dtype=np.float64)

    if np.any(std_i == 0) or np.any(std_q == 0):
        raise ValueError("Cannot auto-process I/Q channel with zero standard deviation")

    i_norm = (i_values - mean_i) / std_i
    q_norm = (q_values - mean_q) / std_q
    sin_inbal = (i_norm * q_norm).mean(axis=0, dtype=np.float64)
    sin_inbal = np.clip(sin_inbal, -0.999999, 0.999999)

    denom = np.sqrt(1.0 - sin_inbal**2)
    i_rot = (i_norm - q_norm * sin_inbal) / denom
    inbalance_deg = np.arcsin(sin_inbal) * 180.0 / np.pi

    complex_echo = (i_rot.astype(np.float32) + 1j * q_norm.astype(np.float32)).astype(np.complex64)
    stats = {
        "mean_i": mean_i.astype(np.float32),
        "mean_q": mean_q.astype(np.float32),
        "std_i": std_i.astype(np.float32),
        "std_q": std_q.astype(np.float32),
        "inbalance_deg": inbalance_deg.astype(np.float32),
    }
    return complex_echo, stats


def extract_polarization(nc: NetCDFClassicFile, pol: str) -> tuple[Any, dict[str, Any]]:
    adc = nc.read_variable("adc_data")
    if adc.ndim != 4:
        raise ValueError(f"Expected 4-D adc_data for alternating TX polarization, got shape {adc.shape}")

    tx_index = 0 if pol[0] == "h" else 1
    receive_like = pol[0] == pol[1]

    like_i = int(nc.read_variable("adc_like_I"))
    like_q = int(nc.read_variable("adc_like_Q"))
    cross_i = int(nc.read_variable("adc_cross_I"))
    cross_q = int(nc.read_variable("adc_cross_Q"))
    i_adc, q_adc = (like_i, like_q) if receive_like else (cross_i, cross_q)

    i_raw = adc[:, tx_index, :, i_adc]
    q_raw = adc[:, tx_index, :, q_adc]
    return auto_process_iq(i_raw, q_raw)


def make_windows(echo: Any, start: int, stop: int, window: int, stride: int) -> Any:
    import numpy as np

    split = echo[start:stop]
    if split.shape[0] < window:
        return np.empty((0, window, split.shape[1]), dtype=np.complex64)

    n_windows = 1 + (split.shape[0] - window) // stride
    shape = (n_windows, window, split.shape[1])
    strides = (split.strides[0] * stride, split.strides[0], split.strides[1])
    return np.lib.stride_tricks.as_strided(split, shape=shape, strides=strides).copy()


def save_split(
    out_path: Path,
    windows: Any,
    labels: Any,
    roles: Any,
    ranges_m: Any,
    stats: dict[str, Any],
    source_file: str,
    pol: str,
    split_name: str,
    entry: dict[str, Any],
    window: int,
    stride: int,
) -> dict[str, Any]:
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_range = np.broadcast_to(labels, (windows.shape[0], labels.shape[0])).copy()
    np.savez_compressed(
        out_path,
        E=windows,
        y_range=y_range,
        range_labels=labels,
        range_roles=roles,
        range_m=ranges_m.astype(np.float32),
        primary_range_bin=np.array(entry["primary"], dtype=np.int16),
        secondary_range_bins=np.array(entry["secondary"], dtype=np.int16),
        mean_i=stats["mean_i"],
        mean_q=stats["mean_q"],
        std_i=stats["std_i"],
        std_q=stats["std_q"],
        inbalance_deg=stats["inbalance_deg"],
        source_file=np.array(source_file),
        polarization=np.array(pol),
        split=np.array(split_name),
        window=np.array(window, dtype=np.int16),
        stride=np.array(stride, dtype=np.int16),
    )

    return {
        "path": str(out_path),
        "source_file": source_file,
        "polarization": pol,
        "split": split_name,
        "windows": int(windows.shape[0]),
        "shape": list(windows.shape),
        "primary": int(entry["primary"]),
        "secondary": [int(item) for item in entry["secondary"]],
    }


def preprocess(args: argparse.Namespace) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: numpy. Install with `python -m pip install -r requirements.txt`."
        ) from exc

    raw_dir = args.raw_dir
    labels = load_labels(args.labels)
    cdf_paths = sorted(raw_dir.glob("*.cdf"))
    if not cdf_paths:
        raise SystemExit(f"No .cdf files found under {raw_dir}")

    missing = [path.name for path in cdf_paths if path.name not in labels]
    if missing:
        raise SystemExit(f"Missing target labels for: {', '.join(missing)}")

    output_root = args.output_dir / f"window{args.window}_stride{args.stride}_{args.target_policy}"
    records: list[dict[str, Any]] = []

    for cdf_path in cdf_paths:
        nc = NetCDFClassicFile(cdf_path)
        adc_var = nc.variables["adc_data"]
        nsweep, ntxpol, nrange, nadc = nc.shape_of(adc_var)
        if (ntxpol, nrange, nadc) != (2, 14, 4):
            raise ValueError(f"Unexpected adc_data shape for {cdf_path.name}: {nc.shape_of(adc_var)}")

        entry = labels[cdf_path.name]
        range_labels, range_roles = make_range_labels(entry, nrange, args.target_policy)
        ranges_m = nc.read_variable("range")
        split_at = int(nsweep * args.train_fraction)

        for pol in args.pols:
            echo, stats = extract_polarization(nc, pol)
            for split_name, start, stop in (
                ("train", 0, split_at),
                ("test", split_at, nsweep),
            ):
                windows = make_windows(echo, start, stop, args.window, args.stride)
                out_name = f"{cdf_path.stem}__{pol}__{split_name}.npz"
                records.append(
                    save_split(
                        output_root / out_name,
                        windows,
                        range_labels,
                        range_roles,
                        ranges_m,
                        stats,
                        cdf_path.name,
                        pol,
                        split_name,
                        entry,
                        args.window,
                        args.stride,
                    )
                )
                print(f"saved {out_name}: E{tuple(windows.shape)}")

    manifest = {
        "raw_dir": str(raw_dir),
        "labels": str(args.labels),
        "output_dir": str(output_root),
        "train_fraction": args.train_fraction,
        "window": args.window,
        "stride": args.stride,
        "target_policy": args.target_policy,
        "polarizations": args.pols,
        "records": records,
        "total_windows": int(sum(item["windows"] for item in records)),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")


def dry_run(args: argparse.Namespace) -> None:
    labels = load_labels(args.labels)
    cdf_paths = sorted(args.raw_dir.glob("*.cdf"))
    if not cdf_paths:
        raise SystemExit(f"No .cdf files found under {args.raw_dir}")

    print("file, adc_shape, primary, secondary, collection_date")
    for cdf_path in cdf_paths:
        nc = NetCDFClassicFile(cdf_path)
        if cdf_path.name not in labels:
            raise SystemExit(f"Missing target labels for {cdf_path.name}")
        adc_shape = nc.shape_of(nc.variables["adc_data"])
        entry = labels[cdf_path.name]
        print(
            f"{cdf_path.name}, {adc_shape}, {entry['primary']}, "
            f"{entry['secondary']}, {nc.attrs.get('Data_collection_date', '')}"
        )


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="Preprocess IPIX Dartmouth CDF files for ST-GNN experiments.")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--raw-dir", type=Path, default=Path(get_config_value(config, "paths.raw_dir")))
    parser.add_argument("--labels", type=Path, default=Path(get_config_value(config, "paths.labels")))
    parser.add_argument("--output-dir", type=Path, default=Path(get_config_value(config, "paths.processed_dir")))
    parser.add_argument(
        "--pols",
        nargs="+",
        choices=POLARIZATIONS,
        default=get_config_value(config, "ipix.polarizations"),
    )
    parser.add_argument("--window", type=int, default=get_config_value(config, "ipix.window"))
    parser.add_argument("--stride", type=int, default=get_config_value(config, "ipix.stride"))
    parser.add_argument("--train-fraction", type=float, default=get_config_value(config, "ipix.train_fraction"))
    parser.add_argument(
        "--target-policy",
        choices=("related", "primary"),
        default=get_config_value(config, "ipix.target_policy"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(remaining)

    if args.window <= 0 or args.stride <= 0:
        raise SystemExit("--window and --stride must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        preprocess(args)


if __name__ == "__main__":
    main()
