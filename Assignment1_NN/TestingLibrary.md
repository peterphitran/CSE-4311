# XOR Classification

Trained a neural network with two inputs, one hidden layer containing two nodes, and one sigmoid output. I tested sigmoid and hyperbolic tangent (Tanh) as the hidden-layer activation functions using the inputs `[[0, 0], [0, 1], [1, 0], [1, 1]]` and XOR targets `[0, 1, 1, 0]`.

After evaluating with a lr of 1.0 and loss < 0.1 both converged and classifying the proper XOR targets
 Sigmoid | Epochs: 619 | Predictions: [0.05769934 0.89873113 0.8986401  0.11718305]
 Tanh | Epochs: 123 | Predictions: [0.09698603 0.93104691 0.9302425  0.13624458]

When trying to evaluate with lr of 0.1 and loss < 0.3 only Tanh converged and clasified proper XOR targets 
 Sigmoid | Epochs: 1000 (capped at 1000) | Predictions: N/A
 Tanh | Epochs: 858 | Predictions: [0.17180053 0.7821323  0.8004907  0.41695467]


While both activations learned the correct XOR classes at higher lr Tanh was easier to train across the board it converged both times and when lr was 1.0 it beat out Sigmoid with only 123 epochs, compared with 619 epochs for sigmoid.
