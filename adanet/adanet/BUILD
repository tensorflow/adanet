# Description:
# A reference implementation of AdaNet as a TensorFlow library.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "adanet",
    srcs = ["__init__.py"],
    visibility = [
    ],
    deps = [
        ":base_learner",
        ":ensemble",
        ":estimator",
        ":evaluator",
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
        ":summary",
        ":timer",
        "@six_archive//:six",
    ],
)

py_test(
    name = "estimator_test",
    size = "large",
    srcs = ["estimator_test.py"],
    shard_count = 5,
    deps = [
        ":base_learner",
        ":estimator",
        ":evaluator",
        ":testing_utils",
        "@absl_py//absl/flags",
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
        ":summary",
    ],
)

py_test(
    name = "iteration_test",
    srcs = ["iteration_test.py"],
    deps = [
        ":base_learner",
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
        "@absl_py//absl/flags",
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
        ":candidate",
        ":ensemble",
    ],
)
