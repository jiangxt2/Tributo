"""Tests for FrozenDict, deep_freeze, and deep_thaw."""

from __future__ import annotations

import json
import pickle
from enum import Enum

import pytest

from tributo._common.immutable import FrozenDict, deep_freeze, deep_thaw


class Color(Enum):
    RED = 1
    GREEN = 2


# ---------------------------------------------------------------------------
# FrozenDict — construction
# ---------------------------------------------------------------------------


class TestFrozenDictConstruction:
    def test_from_empty(self) -> None:
        fd = FrozenDict()
        assert dict(fd) == {}

    def test_from_mapping(self) -> None:
        fd = FrozenDict({"a": 1, "b": 2})
        assert fd["a"] == 1
        assert fd["b"] == 2
        assert len(fd) == 2

    def test_from_kwargs(self) -> None:
        fd = FrozenDict(a=1, b="hello")
        assert fd["a"] == 1
        assert fd["b"] == "hello"

    def test_from_pairs(self) -> None:
        fd = FrozenDict([("x", 10), ("y", 20)])
        assert fd["x"] == 10
        assert fd["y"] == 20

    def test_from_mapping_and_kwargs(self) -> None:
        fd = FrozenDict({"a": 1}, b=2)
        assert fd["a"] == 1
        assert fd["b"] == 2

    def test_nested_mapping_is_recursively_frozen(self) -> None:
        fd = FrozenDict({"a": {"b": 1}})
        inner = fd["a"]
        assert isinstance(inner, FrozenDict)
        assert inner["b"] == 1

    def test_nested_list_is_recursively_frozen(self) -> None:
        fd = FrozenDict({"a": [1, 2, {"b": 3}]})
        outer = fd["a"]
        assert isinstance(outer, tuple)
        assert outer[0] == 1
        assert isinstance(outer[2], FrozenDict)

    def test_non_string_key_raises_in_mapping_mode(self) -> None:
        with pytest.raises(TypeError, match="string"):
            FrozenDict({1: "value"})  # type: ignore[dict-item]

    def test_non_string_key_raises_in_pair_mode(self) -> None:
        with pytest.raises(TypeError, match="string"):
            FrozenDict([(1, "value")])  # type: ignore[list-item]

    def test_non_string_key_raises_via_deep_freeze(self) -> None:
        with pytest.raises(TypeError):
            deep_freeze({1: "value"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# FrozenDict — JSON / pickle compatibility
# ---------------------------------------------------------------------------


class TestFrozenDictSerialization:
    def test_json_roundtrip(self) -> None:
        fd = FrozenDict({"a": 1, "b": [2, 3], "c": {"d": 4}})
        dumped = json.dumps(fd)
        loaded = json.loads(dumped)
        assert loaded == {"a": 1, "b": [2, 3], "c": {"d": 4}}

    def test_json_nested_frozendict_is_just_dict(self) -> None:
        """json does not distinguish FrozenDict from dict."""
        fd = FrozenDict({"outer": FrozenDict({"inner": 42})})
        assert json.dumps(fd) == json.dumps({"outer": {"inner": 42}})

    def test_pickle_roundtrip(self) -> None:
        fd = FrozenDict({"a": 1, "b": ("immutable",)})
        loaded = pickle.loads(pickle.dumps(fd))
        assert loaded == fd
        assert isinstance(loaded, FrozenDict)

    def test_pickle_preserves_frozen_type(self) -> None:
        fd = FrozenDict({"key": [1, 2]})
        loaded = pickle.loads(pickle.dumps(fd))
        assert isinstance(loaded, FrozenDict)
        # pickled mutable containers come back mutable, but the top-level is FrozenDict
        assert isinstance(loaded["key"], tuple)


# ---------------------------------------------------------------------------
# FrozenDict — immutability
# ---------------------------------------------------------------------------


class TestFrozenDictImmutability:
    def test_setitem_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd["a"] = 2

    def test_delitem_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            del fd["a"]

    def test_clear_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.clear()

    def test_pop_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.pop("a")

    def test_popitem_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.popitem()

    def test_setdefault_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.setdefault("b", 2)

    def test_update_raises(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.update({"b": 2})

    def test_copy_returns_mutable(self) -> None:
        fd = FrozenDict({"a": 1})
        mutable = fd.copy()
        assert type(mutable) is dict
        assert mutable == {"a": 1}
        mutable["b"] = 2  # does not raise


# ---------------------------------------------------------------------------
# FrozenDict — hashable
# ---------------------------------------------------------------------------


class TestFrozenDictHash:
    def test_hash_equal_for_equal_contents(self) -> None:
        """Not that FrozenDict needs hashing, but __hash__ exists."""
        a = FrozenDict({"x": 1, "y": 2})
        b = FrozenDict({"y": 2, "x": 1})
        assert hash(a) == hash(b)

    def test_cache_in_set(self) -> None:
        """FrozenDict can be used as a set/dict key."""
        fd = FrozenDict({"a": 1})
        d: dict[FrozenDict, str] = {fd: "value"}
        assert d[FrozenDict({"a": 1})] == "value"


# ---------------------------------------------------------------------------
# deep_freeze
# ---------------------------------------------------------------------------


class TestDeepFreeze:
    def test_primitives_pass_through(self) -> None:
        assert deep_freeze(None) is None
        assert deep_freeze(42) == 42
        assert deep_freeze("hello") == "hello"
        assert deep_freeze(True) is True
        assert deep_freeze(3.14) == 3.14

    def test_enum_passes_through(self) -> None:
        assert deep_freeze(Color.RED) is Color.RED

    def test_mapping_becomes_frozendict(self) -> None:
        result = deep_freeze({"a": 1})
        assert isinstance(result, FrozenDict)
        assert result["a"] == 1

    def test_list_becomes_tuple(self) -> None:
        result = deep_freeze([1, 2, 3])
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_tuple_stays_tuple(self) -> None:
        result = deep_freeze((1, 2, 3))
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_nested_mapping_in_list(self) -> None:
        result = deep_freeze([{"a": 1}, {"b": 2}])
        assert isinstance(result, tuple)
        assert isinstance(result[0], FrozenDict)
        assert isinstance(result[1], FrozenDict)

    def test_nested_list_in_mapping(self) -> None:
        result = deep_freeze({"a": [1, 2]})
        assert isinstance(result, FrozenDict)
        assert isinstance(result["a"], tuple)
        assert result["a"] == (1, 2)

    def test_non_string_key_raises(self) -> None:
        with pytest.raises(TypeError, match="string"):
            deep_freeze({1: "value"})  # type: ignore[dict-item]

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported"):
            deep_freeze({object()})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# deep_thaw
# ---------------------------------------------------------------------------


class TestDeepThaw:
    def test_primitives_pass_through(self) -> None:
        assert deep_thaw(None) is None
        assert deep_thaw(42) == 42
        assert deep_thaw("hello") == "hello"
        assert deep_thaw(True) is True
        assert deep_thaw(3.14) == 3.14

    def test_enum_passes_through(self) -> None:
        assert deep_thaw(Color.RED) is Color.RED

    def test_frozendict_becomes_dict(self) -> None:
        result = deep_thaw(FrozenDict({"a": 1}))
        assert type(result) is dict
        assert result == {"a": 1}

    def test_tuple_becomes_list(self) -> None:
        result = deep_thaw((1, 2, 3))
        assert type(result) is list
        assert result == [1, 2, 3]

    def test_list_stays_list(self) -> None:
        result = deep_thaw([1, 2, 3])
        assert type(result) is list
        assert result == [1, 2, 3]

    def test_nested_frozendict_in_tuple(self) -> None:
        result = deep_thaw((FrozenDict({"a": 1}), FrozenDict({"b": 2})))
        assert type(result) is list
        assert type(result[0]) is dict
        assert type(result[1]) is dict

    def test_nested_tuple_in_frozendict(self) -> None:
        result = deep_thaw(FrozenDict({"a": (1, 2)}))
        assert type(result) is dict
        assert type(result["a"]) is list

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported"):
            deep_thaw(object())


# ---------------------------------------------------------------------------
# freeze → thaw round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_dict_roundtrip(self) -> None:
        original = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        assert deep_thaw(deep_freeze(original)) == original

    def test_deep_nesting(self) -> None:
        original = {
            "training": {
                "lr": 0.01,
                "layers": [128, 64, 32],
                "dropout": {"rate": 0.2, "stochastic_depth": True},
            }
        }
        assert deep_thaw(deep_freeze(original)) == original

    def test_with_enum(self) -> None:
        original = {"color": Color.RED, "items": [Color.GREEN, None, 3.14]}
        assert deep_thaw(deep_freeze(original)) == original

    def test_empty_containers(self) -> None:
        # () rounds to [] after freeze→thaw: freeze keeps (), but thaw
        # converts every tuple→list (no way to distinguish source type).
        original = {"a": {}, "b": [], "c": (), "d": None}
        expected = {"a": {}, "b": [], "c": [], "d": None}
        assert deep_thaw(deep_freeze(original)) == expected

    def test_deep_freeze_then_getitem_in_place(self) -> None:
        """Verify that a frozen result cannot be mutated through accessors."""
        frozen = deep_freeze({"a": [1, 2]})
        inner = frozen["a"]
        assert isinstance(inner, tuple)
        # tuple has no append or __setitem__, so this is naturally immutable


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_freeze_frozendict_is_value_equivalent(self) -> None:
        fd = FrozenDict({"a": 1})
        refrozen = deep_freeze(fd)
        assert refrozen == fd
        assert isinstance(refrozen, FrozenDict)
        # Not identity-idempotent — FrozenDict is a Mapping, so deep_freeze
        # wraps it again.  Semantically correct, just creates a new object.

    def test_thaw_thawed_is_idempotent(self) -> None:
        thawed = deep_thaw(FrozenDict({"a": 1}))
        twice = deep_thaw(thawed)
        assert twice == thawed
        # thaw on an already-thawed (mutable) result is a no-op structurally.
        # Not identity-idempotent because thaw always creates new dicts/lists.
