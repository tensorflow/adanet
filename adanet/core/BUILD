# Description:
# AdaNet is a TensorFlow framework for adaptively learn neural networks as a
# dynamic ensemble of subnetworks.
# This is a reference implementation of AdaNet as a TensorFlow library.

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
        ":base_learner",
        ":base_learner_report",
        ":ensemble",
        ":estimator",
        ":evaluator",
        ":report_materializer",
        ":summary",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    deps = [
        ":candidate",
        ":ensemble",
        ":freezer",
        ":input_utils",
        ":iteration",
        ":report_accessor",
        ":summary",
        ":timer",
        "@six_archive//:six",
    ],
)

py_test(
    name = "estimator_test",
    size = "large",
    srcs = ["estimator_test.py"],
    shard_count = 20,
    deps = [
        ":base_learner",
        ":base_learner_report",
        ":ensemble",
        ":estimator",
        ":evaluator",
        ":report_materializer",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "base_learner",
    srcs = ["base_learner.py"],
)

py_test(
    name = "base_learner_test",
    srcs = ["base_learner_test.py"],
    deps = [
        ":base_learner",
        ":testing_utils",
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
        ":base_learner_report",
        ":summary",
    ],
)

py_test(
    name = "iteration_test",
    srcs = ["iteration_test.py"],
    deps = [
        ":base_learner",
        ":base_learner_report",
        ":candidate",
        ":ensemble",
        ":iteration",
        ":testing_utils",
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
    deps = [
        ":evaluator",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "ensemble",
    srcs = ["ensemble.py"],
    deps = [
    ],
)

py_test(
    name = "ensemble_test",
    srcs = ["ensemble_test.py"],
    deps = [
        ":base_learner",
        ":ensemble",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "freezer",
    srcs = ["freezer.py"],
    deps = [
        ":base_learner",
        ":ensemble",
    ],
)

py_test(
    name = "freezer_test",
    srcs = ["freezer_test.py"],
    deps = [
        ":base_learner",
        ":ensemble",
        ":freezer",
        ":testing_utils",
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
    deps = [
        ":summary",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "input_utils",
    srcs = ["input_utils.py"],
    deps = [
    ],
)

py_test(
    name = "input_utils_test",
    srcs = ["input_utils_test.py"],
    deps = [
        ":input_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "timer",
    srcs = ["timer.py"],
)

py_test(
    name = "timer_test",
    srcs = ["timer_test.py"],
    deps = [
        ":timer",
    ],
)

py_library(
    name = "testing_utils",
    testonly = 1,
    srcs = ["testing_utils.py"],
    deps = [
        ":base_learner",
        ":ensemble",
    ],
)

py_library(
    name = "base_learner_report",
    srcs = ["base_learner_report.py"],
    deps = [
        "@six_archive//:six",
    ],
)

py_test(
    name = "base_learner_report_test",
    srcs = ["base_learner_report_test.py"],
    deps = [
        ":base_learner_report",
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
        ":base_learner_report",
        ":report_pb2",
        "@six_archive//:six",
    ],
)

py_test(
    name = "report_accessor_test",
    srcs = ["report_accessor_test.py"],
    deps = [
        ":base_learner_report",
        ":report_accessor",
    ],
)

py_library(
    name = "report_materializer",
    srcs = ["report_materializer.py"],
    deps = [
        ":base_learner_report",
    ],
)

py_test(
    name = "report_materializer_test",
    srcs = ["report_materializer_test.py"],
    deps = [
        ":base_learner_report",
        ":report_materializer",
        ":testing_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)
