licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

colab_binary(
    name = "notebooks",
    deps = [
        "//adanet",
        "//adanet/examples:simple_dnn",
    ],
)

colab_notebook_test(
    name = "customizing_adanet_notebook_test",
    size = "large",
    colab_binary = ":notebooks",
    ipynb = "customizing_adanet.ipynb",
)
