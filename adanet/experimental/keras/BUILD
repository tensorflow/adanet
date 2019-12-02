# Description:
# Main Keras API logic.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "submodel",
    srcs = ["submodel.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
)

py_library(
    name = "ensemble_model",
    srcs = ["ensemble_model.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        ":submodel",
    ],
)

py_library(
    name = "model_search",
    srcs = ["model_search.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
)

py_library(
    name = "testing_utils",
    srcs = ["testing_utils.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_strict_test(
    name = "ensemble_model_test",
    srcs = ["ensemble_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ensemble_model",
        ":submodel",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)
