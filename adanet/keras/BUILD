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
        "//adanet/autoensemble",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    visibility = whitelist,
    deps = [
        "//adanet/core",
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
