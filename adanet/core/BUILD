# Description:
# Core AdaNet logic.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

whitelist = [
    "//adanet:__subpackages__",
]

py_library(
    name = "core",
    srcs = ["__init__.py"],
    visibility = whitelist,
    deps = [
        ":ensemble_builder",
        ":estimator",
        ":evaluator",
        ":report_materializer",
        ":summary",
        ":tpu_estimator",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    deps = [
        ":architecture",
        ":candidate",
        ":ensemble_builder",
        ":iteration",
        ":report_accessor",
        ":summary",
        ":timer",
        "//adanet/core/ensemble",
        "@six_archive//:six",
    ],
)

py_test(
    name = "estimator_test",
    size = "large",
    srcs = ["estimator_test.py"],
    shard_count = 40,
    srcs_version = "PY2AND3",
    deps = [
        ":ensemble_builder",
        ":estimator",
        ":evaluator",
        ":report_materializer",
        ":testing_utils",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "tpu_estimator",
    srcs = ["tpu_estimator.py"],
    deps = [
        ":ensemble_builder",
        ":estimator",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tpu_estimator_test",
    srcs = ["tpu_estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":testing_utils",
        ":tpu_estimator",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "candidate",
    srcs = ["candidate.py"],
    deps = [
    ],
)

py_test(
    name = "candidate_test",
    srcs = ["candidate_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":candidate",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "iteration",
    srcs = ["iteration.py"],
    deps = [
        ":dict_utils",
        "//adanet/core/subnetwork",
        "@six_archive//:six",
    ],
)

py_test(
    name = "iteration_test",
    srcs = ["iteration_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":candidate",
        ":ensemble_builder",
        ":iteration",
        ":summary",
        ":testing_utils",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    deps = [
    ],
)

py_test(
    name = "evaluator_test",
    srcs = ["evaluator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":evaluator",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "ensemble_builder",
    srcs = ["ensemble_builder.py"],
    deps = [
        ":architecture",
        ":dict_utils",
        ":summary",
        "//adanet/core/subnetwork",
    ],
)

py_test(
    name = "ensemble_builder_test",
    srcs = ["ensemble_builder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ensemble_builder",
        ":summary",
        ":testing_utils",
        "//adanet/core/ensemble",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "summary",
    srcs = ["summary.py"],
    deps = [
    ],
)

py_test(
    name = "summary_test",
    srcs = ["summary_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summary",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

py_library(
    name = "timer",
    srcs = ["timer.py"],
)

py_test(
    name = "timer_test",
    srcs = ["timer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":timer",
    ],
)

py_library(
    name = "testing_utils",
    testonly = 1,
    srcs = ["testing_utils.py"],
    deps = [
        ":architecture",
        ":ensemble_builder",
        "//adanet/core/ensemble",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

proto_library(
    name = "report_proto",
    srcs = ["report.proto"],
    visibility = whitelist,
)



load("@protobuf_bzl//:protobuf.bzl", "py_proto_library")
py_proto_library(
    name = "report_pb2",
    srcs = ["report.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    visibility = whitelist,
)

py_library(
    name = "report_accessor",
    srcs = ["report_accessor.py"],
    deps = [
        ":report_pb2",
        "//adanet/core/subnetwork",
        "@six_archive//:six",
    ],
)

py_test(
    name = "report_accessor_test",
    srcs = ["report_accessor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":report_accessor",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "report_materializer",
    srcs = ["report_materializer.py"],
    deps = [
        "//adanet/core/subnetwork",
    ],
)

py_test(
    name = "report_materializer_test",
    srcs = ["report_materializer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":report_materializer",
        ":testing_utils",
        "//adanet/core/subnetwork",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "dict_utils",
    srcs = ["dict_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)

py_test(
    name = "dict_utils_test",
    srcs = ["dict_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dict_utils",
    ],
)

py_library(
    name = "architecture",
    srcs = ["architecture.py"],
    deps = [
        ":architecture_pb2",
    ],
)

py_test(
    name = "architecture_test",
    srcs = ["architecture_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":architecture",
        "@absl_py//absl/testing:parameterized",
    ],
)

proto_library(
    name = "architecture_proto",
    srcs = ["architecture.proto"],
)

py_proto_library(
    name = "architecture_pb2",
    srcs = ["architecture.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
)
