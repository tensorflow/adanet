licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "simple_dnn",
    srcs = ["simple_dnn.py"],
    visibility = [
        "//adanet:__subpackages__",
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

py_library(
    name = "nasnet",
    srcs = ["nasnet.py"],
    visibility = [
    ],
    deps = [
        "//adanet",
        "@tf_slim//research/slim:nasnet",
        "@tf_slim//research/slim:nets_factory",
    ],
)

py_test(
    name = "nasnet_test",
    srcs = ["nasnet_test.py"],
    shard_count = 4,
    deps = [
        ":nasnet",
        "@absl_py//absl/testing:parameterized",
    ],
)
