# Description:
# Main Keras API logic.

licenses(["notice"])  # Apache 2.0

whitelist = [
    "//adanet:__subpackages__",
]

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    visibility = whitelist,
    deps = [
        ":model",
        "//adanet/autoensemble:keras",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    visibility = ["//adanet/autoensemble:__subpackages__"],
    deps = [
        "//adanet/core",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "model_test",
    size = "medium",
    srcs = ["model_test.py"],
    shard_count = 4,
    deps = [
        ":model",
        "//adanet/core",
        "//adanet/core:testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)
