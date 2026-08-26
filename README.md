# Keras Functional API: Multi-Input and Multi-Output Deep Learning

A practical implementation and demonstration of the **Keras Functional API** for building flexible neural network architectures with multiple inputs, multiple outputs, branching, shared layers, and multi-task learning.

This repository progresses from a basic Functional API model to more advanced architectures involving multiple input tensors and multiple prediction heads.

The primary objective is to understand how the Functional API represents neural networks as **directed computational graphs** rather than restricting models to a simple linear sequence of layers.

---

## Table of Contents

- [Overview](#overview)
- [Why the Functional API](#why-the-functional-api)
- [Repository Structure](#repository-structure)
- [Learning Progression](#learning-progression)
- [Functional API Architecture](#functional-api-architecture)
- [Notebook 1: Basic Functional API](#notebook-1-basic-functional-api)
- [Notebook 2: Multiple Inputs](#notebook-2-multiple-inputs)
- [Notebook 3: Multiple Outputs](#notebook-3-multiple-outputs)
- [Multi-Task Learning](#multi-task-learning)
- [Multi-Output Model Architecture](#multi-output-model-architecture)
- [Loss Functions](#loss-functions)
- [Loss Weighting](#loss-weighting)
- [Training Pipeline](#training-pipeline)
- [Generator Wrapper](#generator-wrapper)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Training Curves](#training-curves)
- [Key Observations](#key-observations)
- [Functional API vs Sequential API](#functional-api-vs-sequential-api)
- [Core Concepts Demonstrated](#core-concepts-demonstrated)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)
- [Future Extensions](#future-extensions)
- [Learning Outcomes](#learning-outcomes)
- [Conclusion](#conclusion)

---

## Overview

The **Keras Functional API** provides a flexible way to construct neural networks by explicitly defining the connections between layers and tensors.

Unlike the Sequential API, where layers are generally arranged in a single linear stack, the Functional API allows neural networks to be represented as arbitrary directed computational graphs.

This makes it suitable for architectures involving:

- Multiple inputs
- Multiple outputs
- Shared layers
- Branching networks
- Merging branches
- Skip connections
- Encoder-decoder architectures
- Siamese networks
- Multi-task learning
- Complex neural network graphs

This repository demonstrates these concepts progressively.

---

## Why the Functional API?

A Sequential model is naturally suited to a simple linear architecture:

```text
Input
  |
  v
Layer 1
  |
  v
Layer 2
  |
  v
Layer 3
  |
  v
Output
```

However, many practical deep learning architectures are not strictly linear.

A model may need to:

1. Accept multiple inputs.
2. Process different inputs through different branches.
3. Combine learned representations.
4. Share layers between multiple tasks.
5. Produce multiple predictions.
6. Use different losses for different outputs.

The Functional API is designed for these situations.

---

## Repository Structure

```text
Functional-api-demo/
│
├── functional-api-demo.ipynb
├── functional-api-multiple-input.ipynb
├── functional-api-multiple-output.ipynb
└── README.md
```

### Notebook Overview

| Notebook | Main Concept |
|---|---|
| `functional-api-demo.ipynb` | Basic Keras Functional API |
| `functional-api-multiple-input.ipynb` | Multiple input tensors |
| `functional-api-multiple-output.ipynb` | Multiple prediction heads and multi-task learning |

---

## Learning Progression

```text
Basic Functional API
        |
        v
Multiple Inputs
        |
        v
Multiple Outputs
        |
        v
Multi-Task Learning
        |
        v
Shared Representations
        |
        v
Task-Specific Prediction Heads
```

---

# Functional API Architecture

At the core of the Functional API are tensors and the connections between them.

A basic Functional API model can be represented as:

```text
Input Tensor
     |
     v
Dense Layer
     |
     v
Dense Layer
     |
     v
Output Tensor
```

The model is then created by specifying the input and output tensors:

```python
inputs = Input(...)
x = Dense(...)(inputs)
x = Dense(...)(x)
outputs = Dense(...)(x)

model = Model(inputs=inputs, outputs=outputs)
```

The important difference is that the architecture is defined through **tensor connectivity**.

---

# Notebook 1: Basic Functional API

## `functional-api-demo.ipynb`

The first notebook introduces the basic Functional API workflow.

The architecture follows a simple computational graph:

```text
Input
  |
  v
Dense Layer
  |
  v
Dense Layer
  |
  v
Output
```

The general flow is:

```text
Input Tensor
     |
     v
Layer
     |
     v
Layer
     |
     v
Output Tensor
     |
     v
Keras Model
```

### Main concepts

- `Input()`
- Functional layer calls
- Tensor flow
- `Model(inputs=..., outputs=...)`
- Model compilation
- Model training
- Prediction

This notebook establishes the foundation for the more complex architectures used later.

---

# Notebook 2: Multiple Inputs

## `functional-api-multiple-input.ipynb`

The second notebook demonstrates how a Functional API model can accept **multiple independent input tensors**.

Instead of forcing all information into a single input tensor, each input can be represented independently and processed through its own branch.

```text
Input A                    Input B
   |                          |
   v                          v
Branch A                   Branch B
   |                          |
   +------------+-------------+
                |
                v
        Feature Fusion
                |
                v
        Shared Layers
                |
                v
             Output
```

The general architecture is:

```text
Input A ──> Feature Extraction ──┐
                                │
                                ├──> Combined Representation ──> Prediction
                                │
Input B ──> Feature Extraction ──┘
```

### Main concepts

- Multiple `Input()` tensors
- Independent processing branches
- Feature extraction
- Concatenation / merging
- Shared downstream layers
- Passing multiple inputs to the model

This architecture is useful when different sources of information need to be processed separately before being combined.

---

# Notebook 3: Multiple Outputs

## `functional-api-multiple-output.ipynb`

The third notebook demonstrates one of the most important capabilities of the Functional API:

> A single neural network can produce multiple predictions simultaneously.

The model performs two different tasks:

1. **Age prediction** — regression
2. **Gender prediction** — binary classification

This is an example of **multi-task learning**.

---

# Multi-Task Learning

Multi-task learning allows one neural network to learn multiple related objectives using a shared representation.

Instead of training two completely independent models:

```text
Model A → Age
Model B → Gender
```

the architecture can share a common feature extractor:

```text
                 Shared Model
                      |
             +--------+--------+
             |                 |
             v                 v
        Age Prediction    Gender Prediction
```

This allows both tasks to make use of features learned by the shared network.

---

# Multi-Output Model Architecture

The multi-output model can be represented as:

```text
Input Image
     |
     v
Shared Feature Extraction
     |
     v
Shared Feature Representation
     |
     +--------------------+
     |                    |
     v                    v
  Age Head            Gender Head
     |                    |
     v                    v
Age Prediction      Gender Prediction
```

The important concept is that the two tasks share the learned representation while having separate prediction heads.

---

# Loss Functions

Because the model performs two different tasks, each output has its own loss function.

## Age Loss

The age prediction uses **Mean Absolute Error (MAE)**.

```text
MAE = mean(|actual_age - predicted_age|)
```

Conceptually:

```text
Actual Age
    |
    v
Predicted Age
    |
    v
Absolute Difference
    |
    v
Mean Absolute Error
```

---

## Gender Loss

The gender prediction uses **Binary Cross-Entropy**.

```text
Binary Cross-Entropy
          |
          v
Binary Classification Loss
```

Therefore:

```text
Age Output
    |
    v
MAE

Gender Output
    |
    v
Binary Cross-Entropy
```

---

# Loss Weighting

The model uses different weights for the two losses.

The compilation configuration is:

```python
loss_weights={
    "age": 1,
    "gender": 99
}
```

Therefore, the total optimization objective can be represented as:

```text
Total Loss
    =
1 × Age Loss
    +
99 × Gender Loss
```

Mathematically:

```text
L_total = 1L_age + 99L_gender
```

This means the gender classification loss has a much larger contribution to the total optimization objective.

---

# Loss Weighting Architecture

```text
Age Prediction
      |
      v
   Age MAE
      |
      v
Weight = 1
      |
      +-------------------+
                          |
                          v
                   Total Loss
                          ^
                          |
      +-------------------+
      |
Weight = 99
      |
      v
Gender Binary Cross-Entropy
      ^
      |
Gender Prediction
```

The optimizer uses the resulting weighted objective to update the shared model parameters.

---

# Model Compilation

The multi-output model is compiled using separate losses and metrics for each output.

```python
model.compile(
    optimizer="adam",

    loss={
        "age": "mae",
        "gender": "binary_crossentropy"
    },

    metrics={
        "age": "mae",
        "gender": "accuracy"
    },

    loss_weights={
        "age": 1,
        "gender": 99
    }
)
```

This demonstrates that each output can have its own:

- Loss function
- Evaluation metric
- Loss weight

---

# Training Pipeline

The complete training process can be represented as:

```text
Training Dataset
       |
       v
Data Generator
       |
       v
Generator Wrapper
       |
       +------------------+
       |                  |
       v                  v
   Input Batch        Target Batch
       |                  |
       +--------+---------+
                |
                v
       Functional API Model
                |
        +-------+-------+
        |               |
        v               v
 Age Prediction   Gender Prediction
        |               |
        v               v
     Age MAE     Binary Cross-Entropy
        |               |
        +-------+-------+
                |
                v
        Weighted Total Loss
                |
                v
          Adam Optimizer
                |
                v
       Parameter Updates
```

---

# Generator Wrapper

The notebook uses a generator wrapper to adapt the generator output into the format expected by the multi-output model.

The wrapper follows this structure:

```python
def generator_wrapper(generator):

    while True:

        x, y = next(generator)

        yield x, tuple(y)
```

The data flow is:

```text
Original Generator
        |
        v
next(generator)
        |
   +----+----+
   |         |
   v         v
Input       Targets
   |         |
   +----+----+
        |
        v
Generator Wrapper
        |
        v
    model.fit()
```

This allows generated batches to be passed cleanly into the multi-output Functional API model.

---

# Training Configuration

The multi-output experiment was trained using:

| Configuration | Value |
|---|---:|
| Optimizer | Adam |
| Epochs | 10 |
| Training Steps / Epoch | 625 |
| Validation Steps | 625 |
| Age Loss | MAE |
| Gender Loss | Binary Cross-Entropy |
| Age Loss Weight | 1 |
| Gender Loss Weight | 99 |

The model tracks separate metrics for both tasks.

### Age

```text
Loss   → MAE
Metric → MAE
```

### Gender

```text
Loss   → Binary Cross-Entropy
Metric → Accuracy
```

---

# Results

The final epoch of the recorded training run produced approximately:

| Metric | Training | Validation |
|---|---:|---:|
| Age MAE | 8.1418 | 7.5123 |
| Gender Accuracy | 82.67% | 86.25% |
| Gender Loss | 0.3728 | 0.3146 |

Therefore, the final validation results were approximately:

```text
Validation Age MAE         ≈ 7.51
Validation Gender Accuracy ≈ 86.25%
Validation Gender Loss     ≈ 0.315
```

These values represent this particular training run and should not be interpreted as a general benchmark for the architecture.

---

# Training Curves

The training logs show that gender classification accuracy improves substantially during the first few epochs and then begins to stabilize.

## Training Gender Accuracy

```text
Epoch 1   0.7402
Epoch 2   0.7891
Epoch 3   0.8082
Epoch 4   0.8128
Epoch 5   0.8159
Epoch 6   0.8220
Epoch 7   0.8235
Epoch 8   0.8249
Epoch 9   0.8271
Epoch 10  0.8267
```

## Validation Gender Accuracy

```text
Epoch 1   0.8355
Epoch 2   0.8482
Epoch 3   0.8125
Epoch 4   0.8534
Epoch 5   0.8563
Epoch 6   0.8608
Epoch 7   0.8641
Epoch 8   0.8724
Epoch 9   0.8468
Epoch 10  0.8625
```

## Training Gender Loss

```text
Epoch 1   0.6967
Epoch 2   0.4433
Epoch 3   0.4117
Epoch 4   0.3999
Epoch 5   0.3951
Epoch 6   0.3820
Epoch 7   0.3819
Epoch 8   0.3793
Epoch 9   0.3719
Epoch 10  0.3728
```

## Validation Gender Loss

```text
Epoch 1   0.3655
Epoch 2   0.3383
Epoch 3   0.3955
Epoch 4   0.3233
Epoch 5   0.3132
Epoch 6   0.3172
Epoch 7   0.3089
Epoch 8   0.2951
Epoch 9   0.3374
Epoch 10  0.3146
```

The plotted training curve in the notebook shows the same overall behavior: training accuracy increases while training loss decreases.

---

# Key Observations

## 1. The Functional API enables non-linear architectures

The model is not restricted to:

```text
Input → Layer → Layer → Output
```

Instead, the graph can branch:

```text
                 +--> Output A
                 |
Input → Shared --+
                 |
                 +--> Output B
```

---

## 2. Multiple tasks can share learned representations

Age and gender prediction use a common feature representation.

```text
Input
  |
  v
Shared Feature Extractor
  |
  v
Shared Representation
  |
  +------> Age
  |
  +------> Gender
```

This is the fundamental idea behind multi-task learning.

---

## 3. Different tasks can use different losses

The age task is a regression problem:

```text
Age → MAE
```

The gender task is a binary classification problem:

```text
Gender → Binary Cross-Entropy
```

The Functional API allows these objectives to coexist inside a single model.

---

## 4. Loss weighting changes the optimization objective

The model uses:

```text
Age Weight = 1
Gender Weight = 99
```

Therefore:

```text
L_total = L_age + 99L_gender
```

The weighting heavily emphasizes the gender classification objective.

This is an important design choice and should not be treated as a neutral configuration.

---

## 5. Validation performance should be interpreted carefully

The final validation gender accuracy is approximately:

```text
86.25%
```

and validation age MAE is approximately:

```text
7.51
```

These results demonstrate learning during the experiment, but they do not establish that the model is production-ready or optimal.

---

# Functional API vs Sequential API

| Feature | Sequential API | Functional API |
|---|---:|---:|
| Linear architectures | Yes | Yes |
| Multiple inputs | No | Yes |
| Multiple outputs | No | Yes |
| Branching | No | Yes |
| Merging branches | No | Yes |
| Skip connections | No | Yes |
| Shared layers | Limited | Yes |
| Complex computational graphs | No | Yes |
| Multi-task learning | Limited | Yes |
| Explicit tensor connections | No | Yes |

A practical rule:

```text
Simple linear architecture
        |
        v
Sequential API
```

For:

```text
Multiple inputs
Multiple outputs
Branches
Merges
Skip connections
Shared layers
        |
        v
Functional API
```

---

# Computational Graph Comparison

## Sequential API

```text
Input
  |
  v
Layer 1
  |
  v
Layer 2
  |
  v
Layer 3
  |
  v
Output
```

The flow is strictly linear.

## Functional API

```text
             +--> Output A
             |
Input --> Shared
             |
             +--> Output B
```

The Functional API allows the architecture to explicitly represent relationships between tensors.

---

# Multi-Input and Multi-Output Architecture

The concepts of multiple inputs and multiple outputs can also be combined.

```text
Input A ──> Processing Branch A ──+
                                  |
                                  v
                           Feature Fusion
                                  |
                                  v
                         Shared Representation
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
                 Output A      Output B      Output C
```

This general architecture appears in many advanced deep learning systems.

---

# Core Concepts Demonstrated

This repository covers:

- Functional model construction
- `Input()` tensors
- Tensor connectivity
- Multiple inputs
- Multiple outputs
- Branching
- Merging
- Shared layers
- Multi-task learning
- Output-specific loss functions
- Output-specific metrics
- Loss weighting
- Generator-based training
- Shared representations
- Training and validation monitoring
- Computational graph design

---

# Mathematical View

For two prediction tasks:

```text
y_age
y_gender
```

the model generates:

```text
ŷ_age
ŷ_gender
```

The individual losses are:

```text
L_age = MAE(y_age, ŷ_age)

L_gender = BinaryCrossEntropy(y_gender, ŷ_gender)
```

The total weighted objective is:

```text
L_total = λ_age L_age + λ_gender L_gender
```

For this experiment:

```text
λ_age = 1
λ_gender = 99
```

Therefore:

```text
L_total = L_age + 99L_gender
```

The optimizer minimizes this combined objective during training.

---

# End-to-End Architecture

```text
Dataset
   |
   v
Data Generator
   |
   v
Generator Wrapper
   |
   v
Input Images
   |
   v
Shared Feature Extraction
   |
   v
Shared Feature Representation
   |
   +--------------------+
   |                    |
   v                    v
Age Head            Gender Head
   |                    |
   v                    v
Age Prediction      Gender Prediction
   |                    |
   v                    v
Age MAE             Binary Cross-Entropy
   |                    |
 Weight 1             Weight 99
   |                    |
   +----------+---------+
              |
              v
      Weighted Total Loss
              |
              v
        Adam Optimizer
              |
              v
   Gradient-Based Updates
              |
              v
        Neural Network
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/harkirat-data/Functional-api-demo.git

cd Functional-api-demo
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib jupyter
```

Depending on the notebook and dataset environment, additional dependencies may be required.

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in this order:

```text
1. functional-api-demo.ipynb
2. functional-api-multiple-input.ipynb
3. functional-api-multiple-output.ipynb
```

---

# Recommended Learning Order

```text
Functional API Fundamentals
          |
          v
Tensor Connections
          |
          v
Multiple Inputs
          |
          v
Multiple Outputs
          |
          v
Multi-Task Learning
          |
          v
Loss Weighting
          |
          v
Complex Neural Network Graphs
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | High-level neural network API |
| Keras Functional API | Computational graph construction |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Visualization |
| Jupyter Notebook | Interactive experimentation |

---

# Limitations

This repository is primarily a learning and experimentation project.

The reported results should not be interpreted as a production benchmark.

Important limitations include:

- The experiment uses only 10 training epochs.
- The architecture is primarily intended to demonstrate Functional API concepts.
- Loss weights are manually selected.
- No systematic hyperparameter search is performed.
- No comparison against independent single-task models is included.
- No production deployment pipeline is included.
- No extensive fairness or subgroup analysis is performed.
- Validation performance alone is insufficient to establish real-world generalization.
- The reported metrics depend on the specific dataset split, preprocessing pipeline, and training configuration.

The primary objective is understanding the architecture and training mechanism rather than achieving state-of-the-art performance.

---

# Future Extensions

## 1. Better Loss Balancing

The current experiment uses:

```text
Age Weight = 1
Gender Weight = 99
```

Future experiments could investigate:

- Dynamic loss weighting
- Uncertainty-based weighting
- Gradient normalization
- Task-specific optimization
- Automated loss balancing

---

## 2. Improved Feature Extractor

The shared backbone could be replaced with a stronger convolutional feature extractor.

```text
Input Image
     |
     v
CNN / Pretrained Backbone
     |
     v
Shared Feature Representation
     |
     +------------+------------+
     |                         |
     v                         v
  Age Head                 Gender Head
     |                         |
     v                         v
Age Prediction           Gender Prediction
```

---

## 3. Additional Prediction Tasks

The same architecture can be extended with additional task-specific heads.

```text
                    Shared Backbone
                           |
       +---------+---------+---------+---------+
       |         |         |         |         |
       v         v         v         v         v
      Age     Gender   Expression Attribute  Other
```

This is one of the main reasons multi-task Functional API architectures are useful.

---

## 4. Experiment With Different Loss Weights

A useful experiment would be to compare:

```text
Age = 1
Gender = 1
```

against:

```text
Age = 1
Gender = 10
```

and:

```text
Age = 1
Gender = 99
```

This would help demonstrate how loss weighting changes the optimization behavior of a multi-task model.

---

## 5. Compare Single-Task vs Multi-Task Models

Another useful experiment would be to train:

```text
Model A → Age only
Model B → Gender only
Model C → Age + Gender
```

and compare their performance.

This would provide a more rigorous evaluation of whether shared representations actually benefit both tasks.

---

# Learning Outcomes

After completing these notebooks, the following concepts should be clear:

- What the Keras Functional API is
- How Functional API models differ from Sequential models
- How tensors define model connectivity
- How to construct multiple-input networks
- How to construct multiple-output networks
- How branching works in neural networks
- How shared representations work
- What multi-task learning means
- How different outputs can use different loss functions
- How output-specific metrics are configured
- How loss weighting affects the optimization objective
- How generators can be integrated with Functional API models
- How to interpret training and validation curves
- How complex neural networks can be represented as computational graphs

---

# Key Takeaway

A neural network does not have to follow a simple:

```text
Input → Layer → Layer → Output
```

structure.

The Functional API allows the model to be represented as a computational graph:

```text
                    +--> Output A
                    |
Input → Shared -----+
                    |
                    +--> Output B
```

This makes it possible to build architectures involving:

```text
Multiple Inputs
       +
Multiple Outputs
       +
Shared Layers
       +
Branching
       +
Merging
       +
Task-Specific Losses
       =
Flexible Deep Learning Architectures
```

The final experiment demonstrates a single neural network learning two tasks simultaneously:

```text
                    Shared Network
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Age Regression        Gender Classification
             |                       |
            MAE              Binary Cross-Entropy
```

The observed validation results from the training run were approximately:

```text
Validation Age MAE         ≈ 7.51
Validation Gender Accuracy ≈ 86.25%
```

The more important outcome is understanding how the Functional API enables these architectures to be constructed and trained within a single computational graph.

---

# Conclusion

This repository provides a progressive introduction to the Keras Functional API.

It begins with a basic Functional API model, moves into multiple-input architectures, and finally demonstrates a multi-output network capable of performing age regression and gender classification simultaneously.

The central concept is that the Functional API provides explicit control over the relationships between tensors and layers.

That flexibility makes it suitable for complex architectures that cannot be represented naturally as simple Sequential models.

```text
Functional API Fundamentals
          |
          v
Multiple Inputs
          |
          v
Multiple Outputs
          |
          v
Shared Representations
          |
          v
Multi-Task Learning
          |
          v
Task-Specific Losses
          |
          v
Loss Weighting
          |
          v
Complex Neural Network Graphs
```

---

# Repository

GitHub Repository:

https://github.com/harkirat-data/Functional-api-demo

---

# Author

**Harkirat Singh**

Data Science / Machine Learning

This repository is part of a hands-on progression through deep learning fundamentals and neural network architectures.
