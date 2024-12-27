import pyirk as p

__URI__ = "irk:/oml_work/1.0"
keymanager = p.KeyManager()
p.register_mod(__URI__, keymanager)
p.start_mod(__URI__)

# TOP-LEVEL CONCEPT
I1001 = p.create_item(
    "I1001",
    R1__has_label="Machine Learning",
    R2__has_description=(
        "A branch of artificial intelligence focused on building applications "
        "that learn from data and improve their performance over time."
    ),
)

# MAJOR SUBFIELDS OF MACHINE LEARNING
I1002 = p.create_item(
    "I1002",
    R1__has_label="Supervised Learning",
    R2__has_description=(
        "A subfield of machine learning that involves training models on labeled "
        "datasets to predict or classify future data."
    ),
)

I1003 = p.create_item(
    "I1003",
    R1__has_label="Unsupervised Learning",
    R2__has_description=(
        "A subfield of machine learning that involves finding patterns in data "
        "without labeled outputs (e.g., clustering, dimensionality reduction)."
    ),
)

I1004 = p.create_item(
    "I1004",
    R1__has_label="Reinforcement Learning",
    R2__has_description=(
        "A subfield of machine learning in which an agent learns to make sequences "
        "of decisions by receiving rewards or penalties."
    ),
)

# COMMON SUPERVISED TASKS
I1005 = p.create_item(
    "I1005",
    R1__has_label="Classification",
    R2__has_description=(
        "A supervised learning task in which the goal is to predict discrete labels, "
        "such as yes/no, spam/not spam, or multi-class categories."
    ),
)

I1006 = p.create_item(
    "I1006",
    R1__has_label="Regression",
    R2__has_description=(
        "A supervised learning task in which the goal is to predict continuous values, "
        "such as prices or temperatures."
    ),
)

# COMMON UNSUPERVISED TASKS

I1007 = p.create_item(
    "I1007",
    R1__has_label="Clustering",
    R2__has_description=(
        "An unsupervised learning task that involves grouping similar data points "
        "based on their features."
    ),
)

I1008 = p.create_item(
    "I1008",
    R1__has_label="Dimensionality Reduction",
    R2__has_description=(
        "An unsupervised approach to reduce the number of input variables while "
        "preserving important structures in the data."
    ),
)

# POPULAR ML ALGORITHMS

# Classification Algorithms
I1010 = p.create_item(
    "I1010",
    R1__has_label="Support Vector Machine (SVM)",
    R2__has_description=(
        "A powerful classification (and sometimes regression) algorithm that finds "
        "a decision boundary (hyperplane) to separate classes with maximal margin."
    ),
)

I1011 = p.create_item(
    "I1011",
    R1__has_label="Decision Tree",
    R2__has_description=(
        "A flowchart-like structure for classification or regression. Each node is a "
        "feature test, branches are outcomes, and leaves are decisions."
    ),
)

I1012 = p.create_item(
    "I1012",
    R1__has_label="Random Forest",
    R2__has_description=(
        "An ensemble method that constructs multiple decision trees at training time "
        "and outputs the class or mean prediction of the individual trees."
    ),
)

I1013 = p.create_item(
    "I1013",
    R1__has_label="Logistic Regression",
    R2__has_description=(
        "A classification algorithm based on the logistic (sigmoid) function, widely "
        "used for binary classification tasks."
    ),
)

# Regression Algorithm
I1014 = p.create_item(
    "I1014",
    R1__has_label="Linear Regression",
    R2__has_description=(
        "A linear approach to modeling the relationship between a scalar response "
        "and one or more explanatory variables."
    ),
)

# Unsupervised Algorithms
I1015 = p.create_item(
    "I1015",
    R1__has_label="K-Means Clustering",
    R2__has_description=(
        "A popular clustering algorithm that partitions the dataset into K clusters "
        "by minimizing the within-cluster sum of squares."
    ),
)

I1016 = p.create_item(
    "I1016",
    R1__has_label="Principal Component Analysis (PCA)",
    R2__has_description=(
        "A dimensionality reduction technique that transforms data to a new coordinate "
        "system, maximizing variance along principal components."
    ),
)

# Reinforcement Learning Algorithm
I1017 = p.create_item(
    "I1017",
    R1__has_label="Q-Learning",
    R2__has_description=(
        "A model-free reinforcement learning algorithm that learns the value (Q-value) "
        "of actions in states by iterative updates based on rewards."
    ),
)

# NEURAL NETWORKS AND RELATED CONCEPTS
I1020 = p.create_item(
    "I1020",
    R1__has_label="Neural Network",
    R2__has_description=(
        "A collection of connected layers of artificial neurons, inspired by biological "
        "neural networks, capable of learning complex functions from data."
    ),
)

I1021 = p.create_item(
    "I1021",
    R1__has_label="Hidden Layer",
    R2__has_description=(
        "A layer of neurons between the input and output layers in a neural network, "
        "enabling more complex representations."
    ),
)

I1022 = p.create_item(
    "I1022",
    R1__has_label="Activation Function",
    R2__has_description=(
        "A non-linear function applied at each neuron to introduce non-linearity, "
        "enabling the network to learn complex patterns."
    ),
)

I1023 = p.create_item(
    "I1023",
    R1__has_label="Multilayer Perceptron (MLP)",
    R2__has_description=(
        "A classic feedforward neural network with multiple layers, typically one or "
        "more hidden layers and a chosen activation function."
    ),
)

I1024 = p.create_item(
    "I1024",
    R1__has_label="Convolutional Neural Network (CNN)",
    R2__has_description=(
        "A deep neural network specialized for grid-like data (e.g., images) using "
        "convolutional and pooling layers to extract spatial features."
    ),
)

# LOSS FUNCTIONS, EVALUATION METRICS, AND OTHER ML CONCEPTS
I1025 = p.create_item(
    "I1025",
    R1__has_label="Loss Function",
    R2__has_description=(
        "A function that quantifies the discrepancy between a model's predictions "
        "and the ground truth, guiding optimization."
    ),
)

I1026 = p.create_item(
    "I1026",
    R1__has_label="Cross Entropy Loss",
    R2__has_description=(
        "A common loss function for classification, measuring the divergence between "
        "two probability distributions."
    ),
)

I1027 = p.create_item(
    "I1027",
    R1__has_label="Mean Squared Error (MSE)",
    R2__has_description=(
        "A popular loss function in regression tasks that averages the squared "
        "differences between predicted and actual values."
    ),
)

I1028 = p.create_item(
    "I1028",
    R1__has_label="Evaluation Metric",
    R2__has_description=(
        "A measure used to assess the performance of a model, e.g., accuracy for "
        "classification or R-squared for regression."
    ),
)

I1029 = p.create_item(
    "I1029",
    R1__has_label="Accuracy",
    R2__has_description=(
        "A classification metric representing the proportion of correctly predicted "
        "instances among all instances."
    ),
)

I1030 = p.create_item(
    "I1030",
    R1__has_label="F1-Score",
    R2__has_description=(
        "The harmonic mean of precision and recall, useful for imbalanced classification."
    ),
)

I1031 = p.create_item(
    "I1031",
    R1__has_label="Precision",
    R2__has_description=(
        "A classification metric measuring the ratio of true positives to all "
        "predicted positives."
    ),
)

I1032 = p.create_item(
    "I1032",
    R1__has_label="Recall",
    R2__has_description=(
        "A classification metric measuring the ratio of true positives to all "
        "actual positives."
    ),
)

I1033 = p.create_item(
    "I1033",
    R1__has_label="R-squared (R2 Score)",
    R2__has_description=(
        "A regression metric representing the proportion of variance in the dependent "
        "variable that is predictable from the independent variables."
    ),
)

I1034 = p.create_item(
    "I1034",
    R1__has_label="Normalization",
    R2__has_description=(
        "A technique that rescales numerical data to a specific range or distribution, "
        "commonly used to stabilize training processes."
    ),
)

# RELATIONS

# Subfields
I1002.set_relation(p.R3["is subclass of"], I1001["Machine Learning"])
I1003.set_relation(p.R3["is subclass of"], I1001["Machine Learning"])
I1004.set_relation(p.R3["is subclass of"], I1001["Machine Learning"])

# Classification/Regression
I1005.set_relation(p.R3["is subclass of"], I1002["Supervised Learning"])
I1006.set_relation(p.R3["is subclass of"], I1002["Supervised Learning"])

# Clustering/DimensionalityReduction
I1007.set_relation(p.R3["is subclass of"], I1003["Unsupervised Learning"])
I1008.set_relation(p.R3["is subclass of"], I1003["Unsupervised Learning"])

# Classification Algorithms
I1010.set_relation(p.R4["is instance of"], I1005["Classification"])
I1011.set_relation(p.R4["is instance of"], I1005["Classification"])
I1012.set_relation(p.R4["is instance of"], I1005["Classification"])
I1013.set_relation(p.R4["is instance of"], I1005["Classification"])

# Regression Algorithm
I1014.set_relation(p.R4["is instance of"], I1006["Regression"])

# Unsupervised Algorithms
I1015.set_relation(p.R4["is instance of"], I1007["Clustering"])
I1016.set_relation(p.R4["is instance of"], I1008["Dimensionality Reduction"])

# Reinforcement Algorithm
I1017.set_relation(p.R4["is instance of"], I1004["Reinforcement Learning"])

# Neural Network
I1020.set_relation(p.R3["is subclass of"], I1001["Machine Learning"])
I1023.set_relation(p.R3["is subclass of"], I1020["Neural Network"])  # MLP
I1024.set_relation(p.R3["is subclass of"], I1020["Neural Network"])  # CNN

# Neural net parts
I1021.set_relation(p.R5["is part of"], I1020["Neural Network"])
I1022.set_relation(p.R5["is part of"], I1020["Neural Network"])

# Loss Functions
I1025.set_relation(p.R5["is part of"], I1001["Machine Learning"])
I1026.set_relation(p.R4["is instance of"], I1025["Loss Function"])
I1027.set_relation(p.R4["is instance of"], I1025["Loss Function"])

# Evaluation Metrics
I1028.set_relation(p.R5["is part of"], I1001["Machine Learning"])
I1029.set_relation(p.R4["is instance of"], I1028["Evaluation Metric"])
I1030.set_relation(p.R4["is instance of"], I1028["Evaluation Metric"])
I1031.set_relation(p.R4["is instance of"], I1028["Evaluation Metric"])
I1032.set_relation(p.R4["is instance of"], I1028["Evaluation Metric"])
I1033.set_relation(p.R4["is instance of"], I1028["Evaluation Metric"])

# Normalization
I1034.set_relation(p.R5["is part of"], I1001["Machine Learning"])


# To be used for depth = 1 visualisation
with open("machine_learning.svg", "w") as f:
    f.write(p.visualize_entity(I1001.uri))

p.end_mod()
# pyirk --load-mod oml.py demo -vis __all__