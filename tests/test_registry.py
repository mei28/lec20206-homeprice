import pytest

from kgl import registry


def test_resolve_dotted_path_returns_the_object():
    import lightgbm

    assert registry.resolve("lightgbm.LGBMClassifier") is lightgbm.LGBMClassifier


def test_resolve_missing_attribute_raises():
    with pytest.raises(AttributeError):
        registry.resolve("lightgbm.NoSuchModel")


def test_resolve_missing_module_raises():
    with pytest.raises(ModuleNotFoundError):
        registry.resolve("no_such_module_xyz.Thing")


def test_register_and_get_roundtrip():
    @registry.register("f_basic")
    def f_basic(x):
        return x + 1

    assert registry.get("f_basic") is f_basic
    assert registry.get("f_basic")(1) == 2


def test_get_unknown_name_raises():
    with pytest.raises(KeyError):
        registry.get("no_such_feature")


def test_build_constructs_with_params():
    from sklearn.model_selection import KFold

    splitter = registry.build("sklearn.model_selection.KFold", n_splits=3, shuffle=True, random_state=0)
    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 3
