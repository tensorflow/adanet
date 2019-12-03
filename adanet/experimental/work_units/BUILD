# Description:
# AdaNet work units.
# **HIGHLY EXPERIMENTAL AND SUBJECT TO CHANGE**

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "work_unit",
    srcs = ["work_unit.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
    ],
)

py_library(
    name = "keras_trainer",
    srcs = ["keras_trainer.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":work_unit",
        "//adanet/experimental/storages:storage",
    ],
)
