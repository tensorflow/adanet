licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "examples",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":simple_dnn",
    ],
)

py_library(
    name = "simple_dnn",
    srcs = ["simple_dnn.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//adanet",
    ],
)

py_test(
    name = "simple_dnn_test",
    srcs = ["simple_dnn_test.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_dnn",
        "@absl_py//absl/testing:parameterized",
    ],
)
