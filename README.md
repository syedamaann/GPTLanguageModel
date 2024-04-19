# Generative Pre-trained Transformer 

## Model Description

The `GPTLanguageModel` leverages a transformer-based architecture, known for its effectiveness in capturing long-range dependencies within data. **Check the Gitlog to track the gradual improvements on the Model**. Key components of the model include:

- **Token and Positional Embeddings**: Converts input tokens into dense vectors that retain positional information.
- **Transformer Blocks**: Composed of multi-head self-attention and position-wise feedforward networks, each block enhances the model's ability to process and relate information across different parts of the input sequence.
- **Layer Normalization**: Applied within each transformer block and after the last block to stabilize the training process by normalizing the activations.
- **Dropout**: Used strategically throughout the model to prevent overfitting by randomly omitting subsets of features during training.

## Key Features

- **Enhanced Hyperparameters**: Adjustments in batch size, block size, learning rate, and architectural dimensions are tuned to balance between computational efficiency and model capacity.
- **Custom Initialization**: Implements specific initializations for linear and embedding layers to ensure stable gradients and effective learning right from the start of training.
- **Robust Training Loop**: Includes detailed monitoring of training and validation loss to gauge performance and convergence throughout the training process.
- **Generation Capabilities**: Equipped with a text generation function that demonstrates the model's ability to produce coherent and contextually appropriate text sequences.

## Installation

To run the model, ensure you have Python 3.6+ and PyTorch 1.7+ installed. Clone this repository, then install the required dependencies:

```bash
pip install torch torchvision
```

## Usage

To train the model, simply run the script from the command line:

```bash
python gpt_language_model.py
```

This will initiate the training process and output the performance metrics during training. Upon completion, the model will generate text based on a predefined seed to demonstrate its capabilities.

## Contributions and Maintenance

Contributions to this project are welcome. You can propose changes by creating a pull request or raise issues if you encounter bugs or have suggestions for improvements.


## Acknowledgments

This project is inspired by Andrej Karpathy's youtube playlist on deep learning and language models. The gradual improvements and implementations reflect a practical approach to learning and applying modern techniques in neural networks and natural language processing.
