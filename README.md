# 🍊 Citrus Quality Classification - Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning project that classifies orange quality into three categories (Good, Medium, Poor) based on physical, chemical, and environmental characteristics. Features a trained logistic regression model with 98.6% accuracy and an interactive web application.

## 📋 Project Overview

This school project demonstrates an end-to-end machine learning pipeline from data analysis to deployment. The system helps farmers and distributors automatically grade orange quality for better market placement and pricing decisions.

## 🎯 Features

- **🤖 Machine Learning Model**: Logistic Regression with 98.6% cross-validation accuracy
- **🛠️ Complete Pipeline**: Integrated data preprocessing, training, and evaluation
- **🌐 Web Application**: Interactive Streamlit app for real-time predictions
- **📊 Data Visualization**: Comprehensive EDA and performance metrics
- **🔧 Production Ready**: Model persistence and easy deployment

## 📁 Dataset

**500 samples** with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `diameter` | Orange diameter (cm) | Numerical |
| `berat` | Weight (grams) | Numerical |
| `tebal_kulit` | Skin thickness (cm) | Numerical |
| `kadar_gula` | Sugar content (%) | Numerical |
| `asal_daerah` | Origin region | Categorical |
| `warna` | Skin color | Categorical |
| `musim_panen` | Harvest season | Categorical |
| `kualitas` | Quality label (Target) | Categorical |

**Quality Classes:**
- 🟢 **Bagus** (Good) - Export quality
- 🟡 **Sedang** (Medium) - Local market quality  
- 🔴 **Jelek** (Poor) - Industrial processing quality

## 🏗️ Model Architecture

```python
Pipeline([
    ('preprocessing', ColumnTransformer([
        ('scaler', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(), categorical_features)
    ])),
    ('model', LogisticRegression())
])
```

## 📈 Performance

| Metric | Score |
|--------|-------|
| **Cross-validation Accuracy** | 98.6% |
| **Test Accuracy** | 100% |
| **Precision** | 99% |
| **Recall** | 99% |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/citrus-quality-classification.git
cd citrus-quality-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app_jeruk.py
```

### Usage Examples

**Train the model:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
```

**Make predictions:**
```python
# Single prediction
new_orange = [[7.5, 180.0, 0.5, 12.0, 'Jawa Barat', 'oranye', 'kemarau']]
prediction = model.predict(new_orange)
probability = model.predict_proba(new_orange)
```

## 🖥️ Web Application

The Streamlit app provides an intuitive interface for quality prediction:

![App Screenshot](https://via.placeholder.com/800x400.png?text=Citrus+Quality+Classifier+App)

**Features:**
- Interactive sliders for numerical features
- Pill selectors for categorical options
- Real-time quality predictions
- Probability visualization
- Business recommendations

## 📂 Project Structure

```
citrus-quality-classification/
│
├── app_jeruk.py                 # Streamlit web application
├── model_klasifikasi_jeruk.joblib # Trained model file
├── jeruk_balance_500.csv        # Dataset
├── requirements.txt             # Dependencies
├── EDA_analysis.ipynb          # Exploratory Data Analysis
└── README.md                   # Project documentation
```

## 🛠️ Technical Stack

- **Programming**: Python 3.8+
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: Joblib

## 📊 Results Analysis

### Feature Importance
The model identifies key factors affecting orange quality:
1. Sugar content (`kadar_gula`)
2. Weight (`berat`) 
3. Skin thickness (`tebal_kulit`)
4. Diameter (`diameter`)

### Business Impact
- **Farmers**: Better pricing decisions based on quality
- **Distributors**: Optimal market channel selection
- **Exporters**: Automated quality control for international standards

## 🔮 Future Enhancements

- [ ] Compare multiple algorithms (Random Forest, SVM, Neural Networks)
- [ ] Add feature importance analysis with SHAP values
- [ ] Develop REST API for integration
- [ ] Mobile app development
- [ ] Real-time image recognition for quality assessment

## 👥 Contributors

- **Your Name** - [GitHub Profile](https://github.com/yourusername)
- School: [Your School Name]
- Course: Machine Learning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided for educational purposes
- Instructors and peers for valuable feedback
- Open-source community for amazing libraries

---

**⭐ If you find this project useful, please give it a star!**

---

### 📞 Contact

For questions or collaborations, feel free to reach out:
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Portfolio: [Your Website](https://yourwebsite.com)

---

<div align="center">
  
**Made with 🍊 and ❤️ for Machine Learning**

</div>
