# Description:
# AdaNet phases.
# **HIGHLY EXPERIMENTAL AND SUBJECT TO CHANGE**

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "phases",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":autoensemble_phase",
        ":input_phase",
        ":keras_trainer_phase",
        ":keras_tuner_phase",
        ":repeat_phase",
    ],
)

py_library(
    name = "phase",
    srcs = ["phase.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [
        "//adanet/experimental/storages:in_memory_storage",
        "//adanet/experimental/storages:storage",
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "keras_trainer_phase",
    srcs = ["keras_trainer_phase.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":phase",
        "//adanet/experimental/storages:in_memory_storage",
        "//adanet/experimental/storages:storage",
        "//adanet/experimental/work_units:keras_trainer_work_unit",
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "keras_tuner_phase",
    srcs = ["keras_tuner_phase.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":phase",
        "//adanet/experimental/work_units:keras_tuner_work_unit",
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "input_phase",
    srcs = ["input_phase.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":phase",
    ],
)

py_library(
    name = "autoensemble_phase",
    srcs = ["autoensemble_phase.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":phase",
        "//adanet/experimental/keras:ensemble_model",
        "//adanet/experimental/storages:in_memory_storage",
        "//adanet/experimental/storages:storage",
        "//adanet/experimental/work_units:keras_trainer_work_unit",
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "repeat_phase",
    srcs = ["repeat_phase.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":phase",
        "//adanet/experimental/work_units:work_unit",
    ],
)
