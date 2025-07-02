# 🚀 AstroExplainr Quick Start Guide

Get **AstroExplainr** up and running in minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## ⚡ Quick Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd AstroExplainr
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the test suite** (optional but recommended)
   ```bash
   python test_app.py
   ```

4. **Start the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in your terminal

## 🎯 First Steps

1. **Select a dataset** from the sidebar
   - Choose from LIGO gravitational wave signals
   - Kepler exoplanet light curves
   - Astronomical images

2. **Explore the analysis**
   - View interactive visualizations
   - Read AI-generated explanations
   - Check summary metrics

3. **Try the comparison feature**
   - Write your own interpretation
   - Compare with AI analysis

## 🔧 Optional Setup

### OpenAI API Integration

For enhanced AI explanations, add your OpenAI API key:

1. **Get an API key** from [OpenAI](https://platform.openai.com/)
2. **Create environment file**
   ```bash
   cp env_example.txt .env
   ```
3. **Edit `.env` file**
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Custom Data Upload

While the current version focuses on sample datasets, you can:
- Modify `data_generators.py` to add your own data
- Extend the file upload functionality in `app.py`

## 🐛 Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Streamlit not found:**
```bash
pip install streamlit
```

**Plotly issues:**
```bash
pip install plotly kaleido
```

**OpenAI API errors:**
- Check your API key is correct
- Ensure you have sufficient credits
- The app will fall back to simulated explanations

### Getting Help

1. **Check the test results**
   ```bash
   python test_app.py
   ```

2. **Review the logs** in your terminal

3. **Check the documentation** in `README.md`

## 🎉 What's Next?

- **Explore different datasets** to see various astrophysics phenomena
- **Experiment with the analysis options** in the sidebar
- **Try writing your own interpretations** and compare with AI
- **Export results** for further analysis

## 📚 Learning Resources

- **Gravitational Waves**: [LIGO Open Science Center](https://www.gw-openscience.org/)
- **Exoplanets**: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- **Astronomy**: [NASA Astrophysics](https://science.nasa.gov/astrophysics)

---

**Happy exploring! 🌌** 