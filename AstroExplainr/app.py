"""
AstroExplainr - Main Streamlit Application
AI-Powered Astrophysics Data Analysis & Explanation Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
import os
from dotenv import load_dotenv

# Import our modules
from data_generators import AstrophysicsDataGenerator, create_sample_datasets
from ml_analysis import analyze_data, AstrophysicsAnalyzer
from llm_explainer import AstrophysicsExplainer, explain_analysis
from visualization import AstrophysicsVisualizer, create_summary_metrics

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AstroExplainr 🌌",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .explanation-box {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #4CAF50;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌌 AstroExplainr</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #888;">AI-Powered Astrophysics Data Analysis & Explanation Tool</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Data Selection")
        
        # Data source selection
        data_source = st.selectbox(
            "Choose Data Source:",
            ["Sample Datasets", "Upload Your Own Data"],
            help="Select from our sample datasets or upload your own astrophysics data"
        )
        
        if data_source == "Sample Datasets":
            # Sample dataset selection
            st.subheader("📊 Sample Datasets")
            
            dataset_category = st.selectbox(
                "Dataset Category:",
                ["Gravitational Waves (LIGO)", "Exoplanets (Kepler)", "Astronomical Images"]
            )
            
            if dataset_category == "Gravitational Waves (LIGO)":
                dataset_name = st.selectbox(
                    "Select LIGO Dataset:",
                    ["Binary Black Hole Merger", "Neutron Star Merger", "Noise/Glitch"]
                )
                dataset_map = {
                    "Binary Black Hole Merger": "ligo_binary_black_hole",
                    "Neutron Star Merger": "ligo_neutron_star", 
                    "Noise/Glitch": "ligo_noise"
                }
                selected_dataset = dataset_map[dataset_name]
                data_type = "ligo"
                
            elif dataset_category == "Exoplanets (Kepler)":
                dataset_name = st.selectbox(
                    "Select Kepler Dataset:",
                    ["Earth-like Planet", "Jupiter-like Planet", "No Transit"]
                )
                dataset_map = {
                    "Earth-like Planet": "kepler_earth_like",
                    "Jupiter-like Planet": "kepler_jupiter_like",
                    "No Transit": "kepler_no_transit"
                }
                selected_dataset = dataset_map[dataset_name]
                data_type = "kepler"
                
            else:  # Astronomical Images
                dataset_name = st.selectbox(
                    "Select Image Dataset:",
                    ["Spiral Galaxy", "Star Cluster", "Empty Field"]
                )
                dataset_map = {
                    "Spiral Galaxy": "galaxy_image",
                    "Star Cluster": "star_cluster_image",
                    "Empty Field": "empty_field_image"
                }
                selected_dataset = dataset_map[dataset_name]
                data_type = "image"
        
        else:  # Upload own data
            st.subheader("📁 Upload Your Data")
            
            uploaded_file = st.file_uploader(
                "Choose a file:",
                type=['csv', 'txt', 'npy', 'npz'],
                help="Upload your astrophysics data file"
            )
            
            if uploaded_file is not None:
                # For now, we'll use a sample dataset as placeholder
                # In a full implementation, you'd parse the uploaded file
                st.info("File upload functionality coming soon! Using sample data for now.")
                selected_dataset = "ligo_binary_black_hole"
                data_type = "ligo"
            else:
                selected_dataset = None
                data_type = None
        
        # Analysis options
        st.header("⚙️ Analysis Options")
        
        show_dashboard = st.checkbox("Show Analysis Dashboard", value=True)
        show_metrics = st.checkbox("Show Summary Metrics", value=True)
        enable_comparison = st.checkbox("Enable Human vs AI Comparison", value=True)
        
        # LLM options
        st.header("🤖 AI Explanation")
        
        use_openai = st.checkbox("Use OpenAI API (if available)", value=True)
        if use_openai and not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI API key not found. Using simulated explanations.")
    
    # Main content area
    if selected_dataset is not None:
        # Load data
        with st.spinner("Loading data..."):
            datasets = create_sample_datasets()
            data = datasets[selected_dataset]
        
        # Run analysis
        with st.spinner("Running ML analysis..."):
            analysis_results = analyze_data(data, data_type)
        
        # Create visualizer
        visualizer = AstrophysicsVisualizer()
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Data Visualization")
            
            # Create appropriate plot
            if data_type == "ligo":
                fig = visualizer.plot_ligo_signal(data, analysis_results)
            elif data_type == "kepler":
                fig = visualizer.plot_kepler_light_curve(data, analysis_results)
            else:  # image
                fig = visualizer.plot_astronomical_image(data, analysis_results)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show dashboard if requested
            if show_dashboard:
                st.subheader("📊 Analysis Dashboard")
                dashboard_fig = visualizer.create_analysis_dashboard(data, analysis_results, data_type)
                st.plotly_chart(dashboard_fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Analysis Summary")
            
            # Show metrics if requested
            if show_metrics:
                metrics = create_summary_metrics(data, analysis_results, data_type)
                
                for metric_name, metric_value in metrics.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{metric_name}:</strong><br>
                        {metric_value}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show original data info
            st.subheader("ℹ️ Data Information")
            if data_type == "ligo":
                st.write(f"**Event Type:** {data['event_info']['type']}")
                if 'mass1' in data['event_info']:
                    st.write(f"**Mass 1:** {data['event_info']['mass1']}")
                    st.write(f"**Mass 2:** {data['event_info']['mass2']}")
                if 'distance' in data['event_info']:
                    st.write(f"**Distance:** {data['event_info']['distance']}")
                st.write(f"**Confidence:** {data['event_info']['confidence']}")
                
            elif data_type == "kepler":
                st.write(f"**Planet Type:** {data['planet_info']['type']}")
                if 'radius' in data['planet_info']:
                    st.write(f"**Radius:** {data['planet_info']['radius']}")
                if 'period' in data['planet_info']:
                    st.write(f"**Period:** {data['planet_info']['period']}")
                if 'transit_depth' in data['planet_info']:
                    st.write(f"**Transit Depth:** {data['planet_info']['transit_depth']}")
                    
            else:  # image
                st.write(f"**Object Type:** {data['object_info']['type']}")
                if 'classification' in data['object_info']:
                    st.write(f"**Classification:** {data['object_info']['classification']}")
                if 'redshift' in data['object_info']:
                    st.write(f"**Redshift:** {data['object_info']['redshift']}")
                if 'magnitude' in data['object_info']:
                    st.write(f"**Magnitude:** {data['object_info']['magnitude']}")
        
        # AI Explanation
        st.subheader("🤖 AI Analysis & Explanation")
        
        with st.spinner("Generating AI explanation..."):
            # Create explainer
            explainer = AstrophysicsExplainer()
            
            # Generate explanation
            if data_type == "ligo":
                ai_explanation = explainer.explain_ligo_analysis(data, analysis_results)
            elif data_type == "kepler":
                ai_explanation = explainer.explain_kepler_analysis(data, analysis_results)
            else:  # image
                ai_explanation = explainer.explain_image_analysis(data, analysis_results)
        
        st.markdown(f"""
        <div class="explanation-box">
            <h4>AI-Generated Explanation:</h4>
            <p>{ai_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Human vs AI Comparison
        if enable_comparison:
            st.subheader("👤 Human vs AI Comparison")
            
            st.write("Write your own interpretation of the data and compare it with the AI's analysis:")
            
            user_explanation = st.text_area(
                "Your Interpretation:",
                placeholder="Describe what you think is happening in this data...",
                height=150,
                help="Share your thoughts on what the data shows"
            )
            
            if user_explanation.strip():
                with st.spinner("Comparing explanations..."):
                    comparison = explainer.compare_explanations(user_explanation, ai_explanation)
                
                st.markdown(f"""
                <div class="comparison-box">
                    <h4>Comparison Analysis:</h4>
                    <p>{comparison}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Technical Details
        with st.expander("🔬 Technical Analysis Details"):
            st.json(analysis_results)
        
        # Data Export
        with st.expander("💾 Export Results"):
            st.write("Export analysis results and visualizations:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Analysis (JSON)"):
                    # Create export data
                    export_data = {
                        "data_type": data_type,
                        "analysis_results": analysis_results,
                        "ai_explanation": ai_explanation,
                        "metadata": data.get("metadata", {}),
                        "event_info": data.get("event_info", data.get("planet_info", data.get("object_info", {})))
                    }
                    
                    # Convert to JSON string
                    import json
                    json_str = json.dumps(export_data, indent=2)
                    
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"astroexplainr_analysis_{data_type}_{selected_dataset}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📈 Export Plot (PNG)"):
                    # Save plot as PNG
                    fig.write_image("astroexplainr_plot.png")
                    
                    with open("astroexplainr_plot.png", "rb") as file:
                        st.download_button(
                            label="Download PNG",
                            data=file.read(),
                            file_name=f"astroexplainr_plot_{data_type}_{selected_dataset}.png",
                            mime="image/png"
                        )
            
            with col3:
                if st.button("📋 Export Summary (CSV)"):
                    # Create summary DataFrame
                    summary_data = []
                    for key, value in create_summary_metrics(data, analysis_results, data_type).items():
                        summary_data.append({"Metric": key, "Value": value})
                    
                    df = pd.DataFrame(summary_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"astroexplainr_summary_{data_type}_{selected_dataset}.csv",
                        mime="text/csv"
                    )
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to AstroExplainr! 🌌
        
        **AstroExplainr** combines machine learning and artificial intelligence to analyze 
        astrophysics data and generate human-readable explanations of astronomical phenomena.
        
        ### 🚀 What You Can Do:
        
        1. **📊 Analyze Data**: Upload your own astrophysics data or explore our sample datasets
        2. **🤖 Get AI Explanations**: Receive natural language explanations of detected phenomena
        3. **📈 Visualize Results**: Interactive plots with highlighted features and anomalies
        4. **👤 Compare Perspectives**: Write your own interpretation and compare with AI analysis
        5. **💾 Export Results**: Download analysis results, plots, and summaries
        
        ### 📋 Supported Data Types:
        
        - **🌊 Gravitational Wave Signals** (LIGO-style)
        - **🪐 Exoplanet Light Curves** (Kepler/TESS-style)  
        - **🖼️ Astronomical Images** (telescope data)
        - **📊 Time Series Data** (general astrophysics signals)
        
        ### 🔬 Analysis Features:
        
        - **ML Pattern Detection**: Automatic identification of peaks, transits, and anomalies
        - **Statistical Analysis**: Comprehensive data statistics and distributions
        - **AI Explanations**: Natural language interpretations using large language models
        - **Interactive Visualizations**: Plotly-powered charts with hover details
        
        ---
        
        **Get started by selecting a dataset from the sidebar!** 🎯
        """)
        
        # Show sample visualizations
        st.subheader("📊 Sample Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Gravitational Wave Signal**")
            # Create sample LIGO plot
            generator = AstrophysicsDataGenerator()
            ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
            ligo_results = analyze_data(ligo_data, "ligo")
            ligo_fig = visualizer.plot_ligo_signal(ligo_data, ligo_results)
            st.plotly_chart(ligo_fig, use_container_width=True)
        
        with col2:
            st.markdown("**Exoplanet Light Curve**")
            # Create sample Kepler plot
            kepler_data = generator.generate_kepler_light_curve("earth_like")
            kepler_results = analyze_data(kepler_data, "kepler")
            kepler_fig = visualizer.plot_kepler_light_curve(kepler_data, kepler_results)
            st.plotly_chart(kepler_fig, use_container_width=True)

if __name__ == "__main__":
    main() 