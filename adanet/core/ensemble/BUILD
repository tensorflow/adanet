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
        "//adanet/core/subnetwork",
    ],
)
