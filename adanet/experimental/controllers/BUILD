# Description:
# AdaNet controller.
# **HIGHLY EXPERIMENTAL AND SUBJECT TO CHANGE**

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "controller",
    srcs = ["controller.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        "//adanet/experimental/storages:storage",
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "sequential_controller",
    srcs = ["sequential_controller.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":controller",
        "//adanet/experimental/phases:phase",
        "//adanet/experimental/storages:in_memory_storage",
        "//adanet/experimental/storages:storage",
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "adanet_controller",
    srcs = ["adanet_controller.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":controller",
        "//adanet/experimental/keras:ensemble_model",
        "//adanet/experimental/phases:phase",
        "//adanet/experimental/storages:in_memory_storage",
        "//adanet/experimental/storages:storage",
        "//adanet/experimental/work_units:keras_trainer",
        "//adanet/experimental/work_units:work_unit",
    ],
)
