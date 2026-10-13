# Zenith

Neural networks for [Saule](https://github.com/lauriszz123/saule). A
define-by-run autograd engine over [numsol](../saule-numsol) arrays, plus
the layers, losses and optimisers that make it a framework rather than a
derivative calculator.

If you know PyTorch, you know this:

```saule
import * from "numsol"
import * from "zenith"

local net: Sequential = Sequential({
  Linear(2, 8),
  Tanh(),
  Linear(8, 1)
})

local opt: Optimizer = Adam(net.parameters(), 0.05)

for epoch: integer = 1, 400 do
  opt.zeroGrad()

  local loss: Tensor = Loss.bceWithLogits(net.forward(inputs), targets)

  loss.backward()
  opt.step()
end
```

## Install

Saule dependencies are **not** transitive, so a project using zenith lists
numsol too:

```
dependencies: ["../saule-zenith", "../saule-numsol"]
```

## How the autograd works

A `Tensor` is an `NDArray` that remembers where it came from. Every
operation on one builds a node recording its inputs and a closure that
knows how to push a gradient back through that single step. `backward()`
walks the graph in reverse topological order and runs those closures, so a
network's gradients fall out of the ordinary forward code — the tape *is*
the program.

Three consequences worth knowing:

- **The graph is rebuilt every forward pass.** Intermediate nodes are
  garbage the moment the step ends. Control flow, loops and branches in the
  forward pass need no special support because there is no static graph to
  disagree with them.
- **Gradients accumulate on leaves**, exactly as in PyTorch. That is what
  makes gradient accumulation across micro-batches free, and why
  `opt.zeroGrad()` comes first in the loop above. Forget it and every step
  sees the sum of every batch so far.
- **Only tensors that need a gradient get a tape.** A forward pass over
  frozen data builds no closures at all, so inference costs what the
  arithmetic costs.

Broadcasting is handled on the way back as well as forward: a bias of shape
`(1, n)` added to a batch of `(m, n)` gets its gradient summed back down
over the batch. That reduction — `unbroadcast` in `tensor.sau` — is the one
piece of the engine that is easy to get subtly wrong, so it is checked
directly.

## Operators

The arithmetic operators build graph nodes, and the right-hand side may be
another tensor, an `NDArray`, a lone `float` or a `table<float>` — anything
that is not already a tensor becomes a constant leaf that takes no
gradient:

```saule
local y: Tensor = (x * 2.0 + 1.0).tanh()
local cost: Tensor = ((y - target) ^ 2.0).mean()
```

As with arrays, arithmetic dispatches on the **left** operand, so the tensor
goes first; `*` is elementwise and `matmul` is the matrix product. `^`
refuses to differentiate with respect to its exponent — that needs
`log(base)`, which is not real for a negative base, so it throws rather
than handing back a quiet NaN.

## What's in it

| Name | Role |
|---|---|
| `Tensor` | an array that remembers its history. Operators, `matmul`, `exp` `log` `sqrt` `tanh` `sigmoid` `relu`, `sum` `mean` `sumAxis` `meanAxis` `softmax`, `reshape` `transpose`, `backward` `gradient` `zeroGrad` `detach` |
| `Module` | the layer contract: `forward(Tensor) -> Tensor` and `parameters() -> table<Tensor>` |
| `Linear` | `input @ weight + bias`, Glorot-uniform initialised |
| `Sequential` | layers applied in order |
| `ReLU` `LeakyReLU` `Tanh` `Sigmoid` `Softmax` | activations |
| `Loss` | `mse` `mae` `crossEntropy` `bceWithLogits` |
| `Metrics` | `accuracy` `binaryAccuracy` `oneHot` |
| `Optimizer` `SGD` `Adam` | parameter updates, applied in place |

`crossEntropy` and `bceWithLogits` take **raw logits** and fuse the softmax
(or sigmoid) into the loss with a hand-written gradient. That is not an
optimisation: composing `log` after `softmax` loses precision and can hand
`log` a zero, while the fused form reduces to `probability - truth`, which
is both exact and cheap.

## Writing your own layer

`Module` is two methods, so a custom layer is a first-class citizen:

```saule
export class Scale implements Module
  local factor: Tensor

  fn init(start: float)
    self.factor = Tensor.parameter(Num.scalar(start), "factor")
  end

  fn forward(input: Tensor) -> Tensor
    return input * self.factor
  end

  fn parameters() -> table<Tensor>
    return {self.factor}
  end
end
```

For a custom *operation* — one with a gradient the chain rule can't reach —
build a node and hand it a closure:

```saule
local out: Tensor = Tensor.node(forwardValue, {input})

out.setBackward(fn()
  input.accumulate(out.gradient() * localDerivative)
end)
```

## Examples

```sh
cd examples/gradcheck && saule run    # 55 checks: autograd vs. finite differences
cd examples/xor       && saule run    # two networks that train
```

**`gradcheck`** nudges every parameter by ±h, re-runs the forward pass, and
compares the central difference against what `backward()` reported — across
every operator, both losses, broadcast gradients, a diamond-shaped graph
and three whole networks. All 55 agree to around 1e-11. A hand-derived
backward pass that is subtly wrong trains to a slightly worse answer rather
than failing loudly, so this is the test that has to pass.

**`xor`** trains the smallest problem a linear model cannot solve
(0.74 → 0.0003, 4/4 correct) and then a three-class softmax classifier on
overlapping Gaussian blobs with SGD, momentum and mini-batches (96.7% on
120 points). Both assert their own outcome, so the example fails rather
than printing a plausible curve that went nowhere.

## Known limits

- **Dense layers only.** No convolution, recurrence, dropout, batch norm or
  attention. `Module` is the extension point and nothing in the engine
  blocks them.
- **No train/eval mode**, because nothing yet behaves differently between
  the two.
- **No checkpointing.** `Linear.setWeight` and `Tensor.setValue` are the
  hooks a save/load pair would be built on.
- **`matmul` is rank 2**, which is numsol's limit too — so no batched
  attention-style products.
- **Speed is interpreter speed.** The engine allocates a fresh array per
  operation; numsol's `kernel.sau` is the one file a native backend would
  replace.
