# ITD105 - Big Data Analytics Lab Exercise #1

## Student Performance Analysis Dashboard

This project implements a comprehensive Exploratory Data Analysis (EDA) dashboard for student exam performance using Streamlit, as part of the ITD105 Big Data Analytics course.

## 📋 Lab Requirements

### Run the program

- python -m streamlit run student_performance.py
### Demo Video
https://github.com/user-attachments/assets/595d445c-e731-4787-9871-46b25779a46a
### Basic Requirements

- [x] Load the dataset into the Streamlit app
- [x] Display the first few rows of the dataset
- [x] Show dataset information (data types, missing values)
- [x] Generate summary statistics for the dataset
- [x] Create a heatmap to visualize correlations between features
- [x] Display a boxplot for exploratory visualization of numeric features
- [x] Use Plotly to create an interactive scatter plot of student performance

### Grade Booster Features

- [x] Interactive widgets for filtering (gender, age, parental education)
- [x] Additional visualization techniques (bar charts, pair plots)
- [x] Dashboard-style layout with tabs, columns, and filtering widgets

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

   ```bash
   git clone <your-repo-url>
   cd Lab1_bigData
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**

   - Download `student-mat.csv` from: https://drive.google.com/drive/folders/1Bz9q37BB20PJSWsdGH__cshZGfPKSpHd?usp=sharing
   - Place the file in the same directory as `student_performance.py`

4. **Run the application**

   ```bash
   streamlit run student_performance.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

## 📊 Features

### Dashboard Tabs

1. **📋 Dataset Info**

   - Dataset overview (rows, columns, memory usage)
   - Data types and null value counts
   - First 10 rows preview

2. **📈 Statistics**

   - Comprehensive summary statistics
   - Mean, median, standard deviation, quartiles

3. **🔥 Correlations**

   - Interactive correlation heatmap
   - Detailed correlation analysis with exam scores
   - Top positive and negative correlations

4. **📊 Visualizations**

   - Customizable boxplots
   - Gender distribution charts
   - Pair plot analysis

5. **🎯 Interactive Analysis**
   - Interactive scatter plots with Plotly
   - Lab questions analysis
   - Gender impact analysis

### Interactive Filters

- **Gender Filter**: Filter data by student gender
- **Age Range**: Slider to filter by age range
- **Parental Education**: Filter by mother's and father's education levels

## 🔍 Lab Questions Analysis

The dashboard automatically analyzes and answers the lab questions:

### a. Highest Correlations with Exam Scores

- Identifies features with strongest correlations to G1, G2, G3
- Shows both positive and negative correlations

### b. Study Time vs Performance

- Calculates correlation between study time and exam performance
- Provides interactive visualization

### c. Boxplot Insights

- Explains what insights can be drawn from boxplot analysis
- Identifies outliers, distribution patterns, and data spread

### d. Gender Impact Analysis

- Compares exam performance between genders
- Shows statistical differences (mean, median, std dev)

## 🛠️ Technical Implementation

### Libraries Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computing

### Key Features

- **Caching**: Data loading is cached for better performance
- **Responsive Design**: Mobile-friendly layout
- **Error Handling**: Graceful handling of missing datasets
- **Custom Styling**: Professional dashboard appearance

## 📁 Project Structure

```
ITD105/
├── student_performance.py    # Main Streamlit application
├── README.md                # This file 
└── student-mat.csv          # Dataset (download separately)
```

## 🎯 Usage Instructions

1. **Start the Application**

   ```bash
   streamlit run student_performance.py
   ```

2. **Navigate the Dashboard**

   - Use the sidebar filters to explore different data subsets
   - Switch between tabs to see different analyses
   - Interact with plots by hovering, zooming, and selecting

3. **Answer Lab Questions**

   - Go to the "🎯 Interactive Analysis" tab
   - Review the automatic analysis of each lab question
   - Use the interactive visualizations to explore further

4. **Record Your Video**
   - Use screen recording software to capture your dashboard
   - Show the key features and analysis results
   - Keep the video under 2 minutes as required

## 📝 Submission Requirements

1. **Source Code**: Submit the `student_performance.py` file
2. **Video Recording**: 2-minute screen recording showing:
   - Dataset loading and basic information
   - Correlation analysis
   - Interactive visualizations
   - Lab questions analysis

## 🔧 Troubleshooting

### Common Issues

1. **Dataset Not Found**

   - Ensure `student-mat.csv` is in the same directory as the Python file
   - Check the file name is exactly `student-mat.csv`

2. **Import Errors**

   - Run `pip install -r requirements.txt` to install all dependencies
   - Ensure you're using Python 3.7 or higher

3. **Port Already in Use**

   - If port 8501 is busy, Streamlit will automatically use the next available port
   - Check the terminal output for the correct URL

4. **Memory Issues**
   - The app uses caching to optimize performance
   - If you encounter memory issues, restart the application

## 🎓 Learning Outcomes

After completing this lab, you will understand:

- How to build interactive data analysis dashboards with Streamlit
- Exploratory Data Analysis techniques and best practices
- Correlation analysis and statistical interpretation
- Data visualization principles and interactive plotting
- Dashboard design and user experience considerations

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## 🤝 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages in the terminal
3. Ensure all dependencies are properly installed
4. Verify the dataset file is correctly placed
