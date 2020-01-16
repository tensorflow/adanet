# Description:
# AdaNet storages.
# **HIGHLY EXPERIMENTAL AND SUBJECT TO CHANGE**

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "storages",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [":in_memory_storage"],
)

py_library(
    name = "storage",
    srcs = ["storage.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [
    ],
)

py_library(
    name = "in_memory_storage",
    srcs = ["in_memory_storage.py"],
    srcs_version = "PY3",
    visibility = ["//adanet:__subpackages__"],
    deps = [
        ":storage",
    ],
)
