# Description:
# Main Keras API logic.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "ensemble_model",
    srcs = ["ensemble_model.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
    ],
)

py_library(
    name = "model_search",
    srcs = ["model_search.py"],
    srcs_version = "PY3",
    visibility = ["//adanet/experimental:__subpackages__"],
    deps = [
        "//adanet/experimental/controllers:controller",
        "//adanet/experimental/schedulers:in_process_scheduler",
        "//adanet/experimental/schedulers:scheduler",
    ],
)

py_strict_test(
    name = "model_search_test",
    srcs = ["model_search_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ensemble_model",
        ":model_search",
        ":testing_utils",
        "//adanet/experimental/controllers:adanet_controller",
        "//adanet/experimental/controllers:sequential_controller",
        "//adanet/experimental/phases:keras_tuner_phase",
        "//adanet/experimental/phases:train_keras_models_phase",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
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
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)
