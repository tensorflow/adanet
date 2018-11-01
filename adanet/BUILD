# Description:
# adanet is a TensorFlow AutoML framework for designing, training, and serving
# adaptive neural network using the AdaNet algorithm.
# This is a reference implementation of AdaNet as a TensorFlow library.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "adanet",
    srcs = [
        "__init__.py",
        "version.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//adanet/core",
    ],
)

py_test(
    name = "adanet_test",
    srcs = ["adanet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":adanet",
        "//adanet/examples",
    ],
)
