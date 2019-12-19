## Expected Prediction and Moments for Probabilistic Circuits
Repository for python implementation and experiments for the paper "On Tractable Computation of Expected Predictions, NeurIPS 2019".

### Repository Structure

- `circuit_expect.py` includes the implementation of the algorithm for computing expectation and moments for pair of probabilistic circuits. This implementation uses `pypsdd` and `LogisticCircuit` libraries for learning and representing the circuits.

- The `./pypsdd` folder includes a copy of the [pypsdd](https://github.com/art-ai/pypsdd) library with some modifications to make it compatible with `Python 3`.

- The `./LogisticCircuit` library includes a copy of the [LogisticCircuit](https://github.com/UCLA-StarAI/LogisticCircuit) library with some additions and modifications to also enable RegressionCircuits.

- The folder `./scripts` include some pyhton scripts to help running the experiments, they range from preprocessing data, learning circuits (psdd, logistic circuit, regression circuit), parallelizing experiments, etc. Additionally, `./scripts/cmd_examples` constains some command ling examples of how to use the scripts.

- The folder `./data` includes the datasets used for the experiments.

- The folder `./exp` includes results such as the learned circuits, and raw results from "missing data experiments".
