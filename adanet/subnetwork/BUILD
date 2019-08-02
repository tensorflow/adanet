# Description:
# Core AdaNet subnetwork logic.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "subnetwork",
    srcs = ["__init__.py"],
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":generator",
        ":report",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
    ],
)

py_test(
    name = "generator_test",
    srcs = ["generator_test.py"],
    deps = [
        ":generator",
        "//adanet/tf_compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "report",
    srcs = ["report.py"],
    deps = [
        "//adanet/tf_compat",
        "@six_archive//:six",
    ],
)

py_test(
    name = "report_test",
    srcs = ["report_test.py"],
    deps = [
        ":report",
        "@absl_py//absl/testing:parameterized",
    ],
)
