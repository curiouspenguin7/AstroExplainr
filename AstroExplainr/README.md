# AstroExplainr 🌌

**AI-Powered Astrophysics Data Analysis & Explanation Tool**

AstroExplainr combines machine learning and large language models to analyze astrophysics data and generate human-readable explanations of astronomical phenomena.

## 🚀 Features

- **Data Upload & Selection**: Upload your own astrophysics data or choose from preset datasets
- **ML Analysis**: Automatic detection of patterns, anomalies, and features in the data
- **AI Explanations**: LLM-powered natural language explanations of detected phenomena
- **Interactive Visualizations**: Plot data with highlighted features and annotations
- **Human vs AI Comparison**: Compare your interpretations with AI-generated explanations

## 📊 Supported Data Types

- **Gravitational Wave Signals** (LIGO-style)
- **Exoplanet Light Curves** (Kepler/TESS-style)
- **Astronomical Images** (telescope data)
- **Time Series Data** (general astrophysics signals)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AstroExplainr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. Run the application:
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Upload Data**: Choose to upload your own CSV/image file or select a preset dataset
2. **Analysis**: The app will automatically run ML analysis to detect features
3. **Explanation**: Review AI-generated explanations of the detected phenomena
4. **Compare**: Write your own interpretation and compare with AI output
5. **Visualize**: Explore interactive plots with highlighted features

## 🔬 Technical Details

- **ML Pipeline**: Uses scikit-learn for anomaly detection and pattern recognition
- **LLM Integration**: OpenAI GPT models for natural language explanations
- **Visualization**: Plotly for interactive charts and matplotlib for static plots
- **Data Processing**: Astropy, GWpy, and Lightkurve for astrophysics data handling

## 📁 Project Structure

```
AstroExplainr/
├── app.py                 # Main Streamlit application
├── data_generators.py     # Simulated data generation
├── ml_analysis.py         # ML analysis pipeline
├── llm_explainer.py       # LLM integration for explanations
├── visualization.py       # Plotting and visualization utilities
├── sample_data/           # Sample datasets
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for the astrophysics and AI communities** 