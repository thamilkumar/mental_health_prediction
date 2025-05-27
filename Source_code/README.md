
# 📌 Project Title:  
**Hyperparameter-Tuned SGD and Gradient Boosting for Mental Health Prediction**

### 🧠 Overview

This project aims to enhance the accuracy of mental health condition predictions using machine learning models like **Stochastic Gradient Descent (SGD)** and **Gradient Boosting (GB)**. The models are optimized with **hyperparameter tuning** and **Recursive Feature Elimination (RFE)** to improve decision-making in mental health support systems.

### 🛠️ Technologies Used

- **Python** (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
- **Machine Learning Models:** SGDClassifier, GradientBoostingClassifier
- **Feature Selection:** Recursive Feature Elimination (RFE)
- **Model Tuning:** GridSearchCV
- **Data Visualization:** Matplotlib, Seaborn
- **Web App Interface:** HTML (optional)

### 📂 Project Structure

```plaintext
mental-health-project/
│
├── dataset/                      # Raw and cleaned datasets
├── model/                        # Saved ML models (pkl or joblib files)
├── notebooks/                    # Jupyter notebooks with EDA and modeling
├── app/                          # Web interface (HTML + Python script)
├── visuals/                      # Graphs and plots
├── README.md                     # Project overview
├── requirements.txt              # Python packages
└── main.py                       # Main training script
```

### 🚀 How to Run the Project

1. Clone the repo:

   ```bash
   git clone https://github.com/thamilkumar/Final_year_project.git
   cd Final_year_project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the model training:

   ```bash
   python main.py
   ```

4. *(Optional)* Launch web interface (if built):

   ```bash
   python app/app.py
   ```

### 📊 Results
- RFE selected optimal features improving model interpretability.
- GridSearchCV improved accuracy by tuning key hyperparameters.
- Gradient Boosting outperformed SGD in precision and recall.

### 📌 Future Work
- Deploy the model via Flask/Streamlit
- Improve dataset size and diversity
- Integrate user feedback loop

### 👨‍💻 Author

**Thamilkumar**  
📧 [thamilkumar003@gmail.com](mailto:thamilkumar003@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/thamilkumar) | [GitHub](https://github.com/thamilkumar)
