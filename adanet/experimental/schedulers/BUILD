# Description:
# AdaNet schedulers.
# **HIGHLY EXPERIMENTAL AND SUBJECT TO CHANGE**

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "schedulers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [":in_process_scheduler"],
)

py_library(
    name = "scheduler",
    srcs = ["scheduler.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "in_process_scheduler",
    srcs = ["in_process_scheduler.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":scheduler",
        "//adanet/experimental/work_units:work_unit",
    ],
)
