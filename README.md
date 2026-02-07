# fredkin_gates_NN
Neural Network Experiments with Fredkin Gates. This is a project where I tried to implement reversible neural networks, meaning that inputs are reconstructible from the output. Currently, it features different implementations of how such layers can look like, with different number of weights per neuron and different combination methods of those weights (e.g. layers consisting of sublayers), inputs, and wiring. Implemented using pyTorch.
Unfortunately, early tests with boolean functions and simple sorting tasks reveal the interdepent nature of weights and thus networks are not properly trainable. Further steps:
- implement alternative intermediate optimizations to reduce interdependence.
- experiemnt with output-to-input wirings and combinations.

Get started using revdifflogic2.ipynb
