# Theory

## Focus on generalization

Generalization error is what we really want to minimize when we train a model.
Most algorithms minimize generalization error indirectly by minimizing a loss
function that consists of a training loss term and additional penalty terms to
discourage the models away from acquiring properties that are associated with
overfitting (e.g., L1 weight norms, L2 weight norms).

## Rigorous trade-offs between training loss and complexity

How do we know what model properties to avoid? Currently, these usually come
from practical experience or industry-accepted best practices. While this has
worked well so far, we would like to minimize the generalization error in a more
principled way.

AdaNet's approach is to minimize a theoretical upper bound on generalization
error, proven in the DeepBoost paper
[[Cortes et al. '14](https://ai.google/research/pubs/pub42856)]:

$$R(f) \leq \widehat{R}_{S, \rho}(f) + \frac{4}{\rho} \sum_{k = 1}^{l} \big \| \mathbf{w}  _k \big \|_1 \mathfrak{R}_m(\widetilde {\cal H}_k) + \widetilde O\Big(\frac{1}{\rho} \sqrt{\frac{\log l}{m}}\Big)$$

This generalization bound allows us to make an apples-to-apples comparison
between the complexities of models in an ensemble and the overall training
loss -- allowing us to design an algorithm that makes this trade-off in a
rigorous manner.

## Other key insights

*   **Convex combinations can't hurt.** Given a set of already-performant and
    uncorrelated base learners, one can take a linear combination of them with
    weights that sum to 1 to obtain an ensemble that outperforms the best among
    those base learners. But even though this ensemble has more trainable
    parameters, it does not have a greater tendency to overfit.
*   **De-emphasize rather than discourage complex models.** If one combines a
    few base learners that are each selected from a different function class
    (e.g., neural networks of different depths and widths), one might expect the
    tendency to overfit to be similar to that of an ensemble comprised of base
    learners selected from the union of all the function classes. Remarkably,
    the DeepBoost bound shows that we can actually do better, as long as the
    final ensemble is a weighted average of model logits where each base
    learner's weight is inversely proportional to the Rademacher complexity of
    its function class, and all the weights in the logits layer sum to 1.
    Additionally, at training time, we don't have to discourage the trainer from
    learning complex models -- it is only when we consider the how much the
    model should contribute to the ensemble do we take the complexity of the
    model into account.
*   **Complexity is not just about the weights.** The Rademacher complexity of a
    neural network does not simply depend on the number of weights or the norm
    of its weights -- it also depends on the number of layers and how they are
    connected. An upper bound on the Rademacher complexity of neural networks
    can be expressed recursively
    [[Cortes et al. '17](https://arxiv.org/abs/1607.01097)], and applies to both
    fully-connected and convolutional neural networks, thus allowing us to
    compute the complexity upper-bounds of almost any neural network that can be
    expressed as a directed-acyclic graph of layers, including unconventional
    architectures such as those found by NASNet
    [[Zoph et al. '17](https://arxiv.org/abs/1707.07012)]. Rademacher complexity
    is also data-dependent, which means that the same neural network
    architecture can have different generalization behavior on different data
    sets.

## AdaNet loss function

Using these insights, AdaNet seeks to minimize the generalization error more
directly using this loss function:

$$\begin{align*} &F\left ( w \right ) = \frac{1}{m} \sum_{i=0}^{N-1} \Phi \left (\sum_{j=0}^{N-1}w_jh_j(x_i), y_i  \right ) + \sum_{j=0}^{N-1} \left (\lambda r(h_j) + \beta   \right )\left | w_j \right |\\ &\text{where }w_j \text{ is the weight of model } j \text{'s contribution to the ensemble,}\\ &h_j \text{ is model } j,\\ &\Phi \text{ is the loss function,}\\ &r(h_j) \text{ is model } j\text{'s complexity, and}\\ &\lambda \text{ and } \beta \text{ are tunable hyperparameters.} \end{align*}$$

By minimizing this loss function, AdaNet is able to combine base learners of
different complexities in a way that generalizes better than one might expect
from the total size of the base learners.


<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.css" integrity="sha384-dbVIfZGuN1Yq7/1Ocstc1lUEm+AT+/rCkibIcC/OmWo5f0EA48Vf8CytHzGrSwbQ" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.js" integrity="sha384-2BKqo+exmr9su6dir+qCw08N2ZKRucY4PrGQPPWU1A7FtlCGjmEGFqXCv5nyM5Ij" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false},
            ]
        });
    });
</script>