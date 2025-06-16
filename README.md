# LearnML - Neural Network for Diabetes Prediction

A simple yet educational implementation of a neural network using PyTorch for binary classification to predict diabetes based on medical data.

## 🎯 Project Overview

This project implements a **SimplePerceptron** neural network that predicts diabetes likelihood using the famous Pima Indian Diabetes dataset. It's designed as a learning tool to understand the fundamentals of neural networks, data preprocessing, and model visualization.

## 🧠 Model Architecture

- **Input Layer**: 8 features (medical indicators)
- **Hidden Layer**: 6 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation (binary classification)

### Features Used
- Pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age

## 📊 Dataset

- **Source**: Pima Indian Diabetes Dataset
- **Samples**: 768 records
- **Features**: 8 medical indicators
- **Target**: Binary classification (0 = No diabetes, 1 = Diabetes)

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch pandas scikit-learn plotly
```

### Running the Project

1. **Train the Model**:
   ```bash
   python "Single Layer/Singlelayer.py"
   ```

2. **Visualize the Network**:
   ```bash
   python "Single Layer/visualize.py"
   ```

## 📁 Project Structure

```
LearnML/
├── Single Layer/
│   ├── Singlelayer.py        # Main neural network implementation
│   ├── visualize.py          # Interactive network visualization
│   ├── pima_diabetes.csv     # Dataset
│   └── trained_model.pth     # Saved model weights
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

## 🔧 Features

- **Educational Focus**: Verbose output and detailed comments for learning
- **Data Preprocessing**: StandardScaler normalization
- **Model Persistence**: Save/load trained models
- **Interactive Visualization**: 2D and 3D network structure visualization
- **Performance Metrics**: Training loss tracking and accuracy evaluation

## 📈 Training Process

The model trains for 50 epochs using:
- **Optimizer**: Adam (learning rate: 0.01)
- **Loss Function**: Binary Cross Entropy
- **Activation Functions**: ReLU (hidden), Sigmoid (output)
- **Data Split**: 80% training, 20% testing

## 🎨 Visualization

Visualizations to help me learn.

## 📊 Expected Results

The model typically achieves:
- Training convergence within 50 epochs
- Test accuracy around 70-80% (varies by random seed)
- Clear visualization of network structure

## 🤝 Contributing

This is an educational project! Feel free to:
- Add more evaluation metrics
- Experiment with different architectures
- Implement additional visualization features
- Add data augmentation techniques

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Repository**: [https://github.com/gray22boyd/LearnML.git](https://github.com/gray22boyd/LearnML.git)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **Dataset Source**: Pima Indian Diabetes Dataset

---

*Created as a learning project to understand neural networks and PyTorch fundamentals.* 