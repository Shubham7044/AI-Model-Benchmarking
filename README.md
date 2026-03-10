# 🚀 AI Model Benchmarking

<p align="center">
A Machine Learning Benchmarking Framework for comparing <b>Classical Machine Learning</b> and <b>Deep Learning models</b> across multiple real-world datasets.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![Framework](https://img.shields.io/badge/Framework-ScikitLearn%20%7C%20TensorFlow%20%7C%20PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

---

# 📖 Overview

The **AI Model Benchmarking Framework** is an automated machine learning experimentation pipeline designed to evaluate and compare multiple algorithms across different datasets.

The framework executes the complete machine learning workflow:

```
Dataset → Preprocessing → Model Training → Evaluation → Leaderboard → Visualization
```

It enables researchers and developers to benchmark models quickly and analyze their performance across different domains.



# 📌 Key Features

✔ Multi-dataset benchmarking \
✔ Classical ML vs Deep Learning comparison\
✔ Automated preprocessing pipeline\
✔ Model leaderboard ranking\
✔ Performance visualization charts\
✔ CSV export of experiment results\
✔ Modular and extensible architecture

---

# 🧠 Models Implemented

| Category      | Algorithm                    |
| ------------- | ---------------------------- |
| Baseline ML   | Logistic Regression          |
| Ensemble ML   | Random Forest                |
| Kernel ML     | Support Vector Machine (SVM) |
| Deep Learning | TensorFlow Neural Network    |
| Deep Learning | PyTorch Neural Network       |

---

# 📂 Datasets Evaluated

| Dataset           | Domain                  | Prediction Task    |
| ----------------- | ----------------------- | ------------------ |
| Heart Disease     | Healthcare              | Disease Prediction |
| Customer Churn    | Business Analytics      | Customer Retention |
| Credit Card Fraud | Finance / Cybersecurity | Fraud Detection    |

---

# 📊 Benchmark Results

## ❤️ Heart Disease Dataset

| Rank | Model                     | Accuracy   |
| ---- | ------------------------- | ---------- |
| 🥇   | Random Forest             | **0.9854** |
| 🥈   | TensorFlow Neural Network | 0.9171     |
| 🥉   | SVM                       | 0.8878     |
| 4    | PyTorch Neural Network    | 0.8878     |
| 5    | Logistic Regression       | 0.7951     |

**Insight**

Tree-based ensemble models like **Random Forest** perform extremely well on structured healthcare datasets.

---

## 📉 Customer Churn Dataset

| Rank | Model                     | Accuracy   |
| ---- | ------------------------- | ---------- |
| 🥇   | Random Forest             | **0.7956** |
| 🥈   | Logistic Regression       | 0.7722     |
| 🥉   | SVM                       | 0.7573     |
| 4    | TensorFlow Neural Network | 0.7012     |
| 5    | PyTorch Neural Network    | 0.6388     |

**Insight**

For tabular business datasets, classical machine learning models often outperform deep learning models.

---

## 💳 Credit Card Fraud Dataset

| Rank | Model                     | Accuracy   |
| ---- | ------------------------- | ---------- |
| 🥇   | Random Forest             | **0.9996** |
| 🥈   | TensorFlow Neural Network | 0.9995     |
| 🥉   | SVM                       | 0.9993     |
| 4    | Logistic Regression       | 0.9991     |
| 5    | PyTorch Neural Network    | 0.9983     |

**Insight**

Fraud detection datasets are highly imbalanced, resulting in extremely high accuracy scores across most models.

---

# ⚙️ System Workflow

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Encoding + Scaling
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Leaderboard
   │
   ▼
Benchmark Charts + CSV Results
```

---

# 🏗️ Project Structure

```
AI-Model-Benchmarking/
│
├── data/
│
├── results/
│   ├── heart_disease_chart.png
│   ├── churn_chart.png
│   ├── fraud_chart.png
│   ├── heart_disease_results.csv
│   ├── churn_results.csv
│   └── fraud_results.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── sklearn_models.py
│   ├── tensorflow_model.py
│   └── pytorch_model.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Model-Benchmarking.git
cd AI-Model-Benchmarking
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Benchmark

Run the full benchmarking pipeline:

```bash
python main.py
```

The system automatically:

1. Load datasets
2. Preprocess data
3. Train ML and DL models
4. Evaluate performance
5. Rank models
6. Generate charts and CSV results

---

# 📁 Output

Generated files will be located in the `results/` directory.

```
results/
├── heart_disease_chart.png
├── churn_chart.png
├── fraud_chart.png
├── heart_disease_results.csv
├── churn_results.csv
└── fraud_results.csv
```

---

# 🔧 Technologies Used

| Category         | Tools               |
| ---------------- | ------------------- |
| Programming      | Python              |
| Machine Learning | Scikit-learn        |
| Deep Learning    | TensorFlow, PyTorch |
| Data Processing  | Pandas, NumPy       |
| Visualization    | Matplotlib          |

---

# 💡 Key Learnings

* Classical ML models often outperform deep learning models on tabular datasets
* Random Forest provides strong performance across multiple domains
* Benchmarking frameworks help with systematic model comparison
* Multi-dataset evaluation improves model selection insights

---

# 🚀 Future Improvements

* Cross-validation benchmarking
* Hyperparameter optimization
* Additional models (XGBoost, LightGBM)
* AutoML benchmarking pipeline
* Experiment tracking dashboard

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```
git checkout -b feature/your-feature-name
```

3. Commit your changes

```
git commit -m "Add feature"
```

4. Push to GitHub

```
git push origin feature/your-feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Shubham Swarnakar**
B.Tech Computer Science (AI & ML)

---

⭐ If you found this project useful, consider **starring the repository**.
