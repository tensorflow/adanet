# Description:
# Auto-ensemble logic.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "autoensemble",
    srcs = ["__init__.py"],
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":common",
        ":estimator",
        ":keras",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//adanet:__subpackages__"],
    deps = [
        "//adanet/subnetwork",
        "//adanet/tf_compat",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    visibility = ["//adanet/keras:__subpackages__"],
    deps = [
        ":common",
        "//adanet/keras:model",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    deps = [
        ":common",
        "//adanet/core",
    ],
)

py_test(
    name = "keras_test",
    size = "medium",
    srcs = ["keras_test.py"],
    deps = [
        ":keras",
        "//adanet/tf_compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "estimator_test",
    size = "large",
    srcs = ["estimator_test.py"],
    shard_count = 5,
    deps = [
        ":estimator",
        "//adanet/tf_compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "tpu_estimator_test",
    size = "large",
    srcs = ["tpu_estimator_test.py"],
    shard_count = 3,
    deps = [
        ":estimator",
        "//adanet/tf_compat",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "estimator_v2_test",
    size = "large",
    srcs = ["estimator_v2_test.py"],
    shard_count = 5,
    deps = [
        ":estimator",
        "@absl_py//absl/testing:parameterized",
    ],
)
