# Description:
# AdaNet ensemble logic.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "ensemble",
    srcs = ["__init__.py"],
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":ensembler",
        ":strategy",
        ":weighted",
    ],
)

py_library(
    name = "ensembler",
    srcs = ["ensembler.py"],
    deps = [
    ],
)

py_library(
    name = "strategy",
    srcs = ["strategy.py"],
    deps = [
    ],
)

py_test(
    name = "strategy_test",
    srcs = ["strategy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ensemble",
        "//adanet/subnetwork",
    ],
)

py_library(
    name = "weighted",
    srcs = ["weighted.py"],
    deps = [
        ":ensembler",
        "//adanet/tf_compat",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "weighted_test",
    srcs = ["weighted_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ensemble",
        "//adanet/core",
        "//adanet/subnetwork",
        "//adanet/tf_compat",
        "@absl_py//absl/testing:parameterized",
    ],
)
