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
        "//adanet/experimental/work_units:work_unit",
    ],
)

py_library(
    name = "static_controller",
    srcs = ["static_controller.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":controller",
        "//adanet/experimental/phases:phase",
        "//adanet/experimental/work_units:work_unit",
    ],
)
