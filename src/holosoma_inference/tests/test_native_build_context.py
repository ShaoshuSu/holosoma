"""Native editable installs need implementation source in the dependency stage."""

import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("filename", "install_marker", "expected_installs"),
    [
        ("docker/Dockerfile", "bash scripts/setup_all.sh", 1),
        ("docker/thor/Dockerfile", "RUN uv pip install -e", 2),
        ("src/holosoma_inference/docker/Dockerfile", "RUN uv pip install --system -e", 1),
    ],
)
def test_every_native_install_has_cmake_sources_in_its_stage(filename, install_marker, expected_installs):
    lines = (ROOT / filename).read_text().replace("\\\n", " ").splitlines()
    copied_native = False
    installs = 0
    for line in lines:
        if line.startswith("FROM "):
            copied_native = False
        if line.startswith("COPY "):
            words = shlex.split(line)
            copied_native |= "src/holosoma_inference/holosoma_cpp" in words[1:-1]
        if install_marker in line:
            installs += 1
            assert copied_native, f"{filename}: native install precedes its CMake source COPY"
    assert installs == expected_installs, "fixture must cover every editable native install"
