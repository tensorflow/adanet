# Description:
# adanet is a TensorFlow AutoML framework for designing, training, and serving
# adaptive neural network using the AdaNet algorithm.
# This is a reference implementation of AdaNet as a TensorFlow library.

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "adanet",
    srcs = [
        "__init__.py",
        "version.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//adanet/autoensemble",
        "//adanet/core",
        "//adanet/distributed",
        "//adanet/ensemble",
        "//adanet/keras",
        "//adanet/replay",
        "//adanet/subnetwork",
    ],
)

py_test(
    name = "adanet_test",
    srcs = ["adanet_test.py"],
    deps = [
        ":adanet",
        "//adanet/examples",
    ],
)
