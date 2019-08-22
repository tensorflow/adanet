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
        ":mean",
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
    deps = [
        ":ensemble",
        "//adanet/core",
        "//adanet/subnetwork",
        "//adanet/tf_compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "mean",
    srcs = ["mean.py"],
    deps = [
        ":ensembler",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "mean_test",
    srcs = ["mean_test.py"],
    deps = [
        ":ensemble",
        "//adanet/core",
        "//adanet/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)
