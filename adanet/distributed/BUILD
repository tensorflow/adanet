# Description:
# AdaNet distributed logic.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "distributed",
    srcs = ["__init__.py"],
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":devices",
        ":placement",
    ],
)

py_library(
    name = "devices",
    srcs = ["devices.py"],
    tags = [
        "PYDEPS=WARN",
    ],
    deps = [
    ],
)

py_test(
    name = "devices_test",
    srcs = ["devices_test.py"],
    deps = [
        ":devices",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "placement",
    srcs = ["placement.py"],
    deps = [
        ":devices",
        "//adanet/tf_compat",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "placement_test",
    srcs = ["placement_test.py"],
    deps = [
        ":placement",
        "@absl_py//absl/testing:parameterized",
    ],
)
