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
    deps = [
        ":model",
        "//adanet/core",
        "//adanet/core:testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)
