licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "simple_dnn",
    srcs = ["simple_dnn.py"],
    visibility = [
    ],
    deps = [
        "//adanet",
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
