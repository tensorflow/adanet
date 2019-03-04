licenses(["notice"])  # Apache 2.0

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        ":adanet_improve_nas",
        ":cifar10",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "improve_nas",
    srcs = ["improve_nas.py"],
    deps = [
        ":nasnet",
        ":subnetwork_utils",
        "//adanet",
    ],
)

py_library(
    name = "adanet_improve_nas",
    srcs = ["adanet_improve_nas.py"],
    deps = [
        ":improve_nas",
        ":optimizer",
        "//adanet",
    ],
)

py_library(
    name = "subnetwork_utils",
    srcs = ["subnetwork_utils.py"],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
)

py_library(
    name = "cifar10",
    srcs = ["cifar10.py"],
    deps = [
        ":image_processing",
    ],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nasnet_utils.py"],
)

py_library(
    name = "image_processing",
    srcs = ["image_processing.py"],
)

py_library(
    name = "nasnet",
    srcs = ["nasnet.py"],
    deps = [
        ":nasnet_utils",
    ],
)
