licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "examples",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":simple_dnn",
    ],
)

py_library(
    name = "simple_dnn",
    srcs = ["simple_dnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//adanet",
        "@six_archive//:six",
    ],
)

py_test(
    name = "simple_dnn_test",
    srcs = ["simple_dnn_test.py"],
    deps = [
        ":simple_dnn",
        "@absl_py//absl/testing:parameterized",
    ],
)
