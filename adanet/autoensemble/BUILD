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
    ],
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//adanet/core",
    ],
)

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        "@absl_py//absl/testing:parameterized",
    ],
)
