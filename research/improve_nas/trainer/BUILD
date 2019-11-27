licenses(["notice"])  # Apache 2.0

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adanet_improve_nas",
        ":cifar10",
        ":fake_data",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "improve_nas",
    srcs = ["improve_nas.py"],
    srcs_version = "PY3",
    deps = [
        ":nasnet",
        ":subnetwork_utils",
        "//adanet",
    ],
)

py_test(
    name = "improve_nas_test",
    srcs = ["improve_nas_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fake_data",
        ":improve_nas",
        "//adanet",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "adanet_improve_nas",
    srcs = ["adanet_improve_nas.py"],
    srcs_version = "PY3",
    deps = [
        ":improve_nas",
        ":optimizer",
        "//adanet",
    ],
)

py_test(
    name = "adanet_improve_nas_test",
    srcs = ["adanet_improve_nas_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adanet_improve_nas",
        ":fake_data",
        "//adanet",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "subnetwork_utils",
    srcs = ["subnetwork_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY3",
)

py_library(
    name = "cifar10",
    srcs = ["cifar10.py"],
    srcs_version = "PY3",
    deps = [
        ":image_processing",
    ],
)

py_test(
    name = "cifar10_test",
    srcs = ["cifar10_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cifar10",
    ],
)

py_library(
    name = "cifar100",
    srcs = ["cifar100.py"],
    srcs_version = "PY3",
    deps = [
        ":image_processing",
    ],
)

py_test(
    name = "cifar100_test",
    srcs = ["cifar100_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cifar100",
    ],
)

py_library(
    name = "fake_data",
    srcs = ["fake_data.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nasnet_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "image_processing",
    srcs = ["image_processing.py"],
    srcs_version = "PY3",
)

py_library(
    name = "nasnet",
    srcs = ["nasnet.py"],
    srcs_version = "PY3",
    deps = [
        ":nasnet_utils",
    ],
)
