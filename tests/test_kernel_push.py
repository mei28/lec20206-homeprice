import pytest

from tools import kernel_push

TEMPLATE = 'CODES = "/kaggle/input/$CODES"\nMODEL = "/kaggle/input/$MODEL"\nEXP = "$EXP"\nDEPS = "$DEPS"\n'


def test_render_fills_every_placeholder():
    out = kernel_push.render(
        TEMPLATE, codes="titanic-codes", model="titanic-exp000-baseline-model", exp="exp000_baseline", deps=""
    )
    assert out == (
        'CODES = "/kaggle/input/titanic-codes"\n'
        'MODEL = "/kaggle/input/titanic-exp000-baseline-model"\n'
        'EXP = "exp000_baseline"\n'
        'DEPS = ""\n'
    )
    assert "$" not in out


def test_kernel_metadata_lists_sources_and_disables_internet():
    meta = kernel_push.kernel_metadata("mei28", "titanic", codes="titanic-codes", model="titanic-exp000-baseline-model")
    assert meta["id"] == "mei28/titanic-sub"
    assert meta["title"] == "titanic-sub"
    assert meta["code_file"] == "sub.ipynb"
    assert meta["kernel_type"] == "notebook" and meta["language"] == "python"
    assert meta["is_private"] is True and meta["enable_internet"] is False and meta["enable_gpu"] is False
    assert meta["dataset_sources"] == ["mei28/titanic-codes", "mei28/titanic-exp000-baseline-model"]
    assert meta["competition_sources"] == ["titanic"]


def test_kernel_metadata_adds_deps_and_gpu_when_asked():
    meta = kernel_push.kernel_metadata("mei28", "titanic", codes="c", model="m", deps="titanic-deps", gpu=True)
    assert meta["dataset_sources"] == ["mei28/c", "mei28/m", "mei28/titanic-deps"]
    assert meta["enable_gpu"] is True


def test_kernel_metadata_rejects_long_titles():
    with pytest.raises(ValueError, match="50"):
        kernel_push.kernel_metadata("mei28", "x" * 60, codes="c", model="m")


def test_parse_version_reads_the_push_output():
    assert kernel_push.parse_version("Kernel version 3 successfully pushed.  Please check progress at https://...") == 3


def test_parse_version_raises_on_unexpected_output():
    with pytest.raises(RuntimeError):
        kernel_push.parse_version("400 Bad Request")
