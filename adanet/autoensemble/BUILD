# Description:
# Auto-ensemble logic.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "autoensemble",
    srcs = ["__init__.py"],
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":estimator",
        ":keras",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        ":estimator",
        "//adanet/keras:model",
    ],
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    deps = [
        "//adanet/core",
        "//adanet/subnetwork",
        "//adanet/tf_compat",
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
    shard_count = 4,
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
