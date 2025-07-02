"""
Visualization Module for AstroExplainr
Creates interactive plots and visualizations for astrophysics data
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional

class AstrophysicsVisualizer:
    """Creates visualizations for astrophysics data"""
    
    def __init__(self):
        """Initialize the visualizer"""
        # Set up matplotlib style
        plt.style.use('dark_background')
        
    def plot_ligo_signal(self, data: Dict, analysis_results: Dict, use_plotly: bool = True):
        """
        Create interactive plot for LIGO gravitational wave signal
        
        Args:
            data (dict): LIGO data
            analysis_results (dict): Analysis results
            use_plotly (bool): Whether to use Plotly (True) or Matplotlib (False)
            
        Returns:
            plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        time = data["time"]
        signal = data["signal"]
        
        if use_plotly:
            # Create Plotly figure
            fig = go.Figure()
            
            # Main signal
            fig.add_trace(go.Scatter(
                x=time,
                y=signal,
                mode='lines',
                name='Gravitational Wave Signal',
                line=dict(color='cyan', width=1),
                hovertemplate='Time: %{x:.3f}s<br>Strain: %{y:.6f}<extra></extra>'
            ))
            
            # Highlight peaks
            if analysis_results.get('peaks'):
                peak_times = [peak['time'] for peak in analysis_results['peaks']]
                peak_amplitudes = [peak['amplitude'] for peak in analysis_results['peaks']]
                
                fig.add_trace(go.Scatter(
                    x=peak_times,
                    y=peak_amplitudes,
                    mode='markers',
                    name='Detected Peaks',
                    marker=dict(color='red', size=8, symbol='diamond'),
                    hovertemplate='Peak at %{x:.3f}s<br>Amplitude: %{y:.6f}<extra></extra>'
                ))
            
            # Highlight anomalies
            if analysis_results.get('anomalies'):
                anomaly_times = [anomaly['time'] for anomaly in analysis_results['anomalies']]
                anomaly_amplitudes = [anomaly['amplitude'] for anomaly in analysis_results['anomalies']]
                
                fig.add_trace(go.Scatter(
                    x=anomaly_times,
                    y=anomaly_amplitudes,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='orange', size=6, symbol='circle'),
                    hovertemplate='Anomaly at %{x:.3f}s<br>Amplitude: %{y:.6f}<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=f"LIGO Gravitational Wave Signal - {data['event_info']['type']}",
                xaxis_title="Time (seconds)",
                yaxis_title="Strain",
                template="plotly_dark",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
        
        else:
            # Create Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Main signal
            ax.plot(time, signal, color='cyan', linewidth=1, label='Gravitational Wave Signal')
            
            # Highlight peaks
            if analysis_results.get('peaks'):
                peak_times = [peak['time'] for peak in analysis_results['peaks']]
                peak_amplitudes = [peak['amplitude'] for peak in analysis_results['peaks']]
                ax.scatter(peak_times, peak_amplitudes, color='red', s=50, marker='d', 
                          label='Detected Peaks', zorder=5)
            
            # Highlight anomalies
            if analysis_results.get('anomalies'):
                anomaly_times = [anomaly['time'] for anomaly in analysis_results['anomalies']]
                anomaly_amplitudes = [anomaly['amplitude'] for anomaly in analysis_results['anomalies']]
                ax.scatter(anomaly_times, anomaly_amplitudes, color='orange', s=30, 
                          label='Anomalies', zorder=5)
            
            ax.set_title(f"LIGO Gravitational Wave Signal - {data['event_info']['type']}")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Strain")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_kepler_light_curve(self, data: Dict, analysis_results: Dict, use_plotly: bool = True):
        """
        Create interactive plot for Kepler light curve
        
        Args:
            data (dict): Kepler data
            analysis_results (dict): Analysis results
            use_plotly (bool): Whether to use Plotly (True) or Matplotlib (False)
            
        Returns:
            plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        time = data["time"]
        flux = data["flux"]
        
        if use_plotly:
            # Create Plotly figure
            fig = go.Figure()
            
            # Main light curve
            fig.add_trace(go.Scatter(
                x=time,
                y=flux,
                mode='lines',
                name='Stellar Flux',
                line=dict(color='yellow', width=1),
                hovertemplate='Time: %{x:.2f}h<br>Flux: %{y:.4f}<extra></extra>'
            ))
            
            # Highlight transits
            if analysis_results.get('transits'):
                transit_times = [transit['time'] for transit in analysis_results['transits']]
                transit_fluxes = [flux[np.argmin(np.abs(time - t))] for t in transit_times]
                
                fig.add_trace(go.Scatter(
                    x=transit_times,
                    y=transit_fluxes,
                    mode='markers',
                    name='Detected Transits',
                    marker=dict(color='red', size=10, symbol='diamond'),
                    hovertemplate='Transit at %{x:.2f}h<br>Depth: %{y:.4f}<extra></extra>'
                ))
            
            # Highlight anomalies
            if analysis_results.get('anomalies'):
                anomaly_times = [anomaly['time'] for anomaly in analysis_results['anomalies']]
                anomaly_fluxes = [anomaly['flux'] for anomaly in analysis_results['anomalies']]
                
                fig.add_trace(go.Scatter(
                    x=anomaly_times,
                    y=anomaly_fluxes,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='orange', size=6, symbol='circle'),
                    hovertemplate='Anomaly at %{x:.2f}h<br>Flux: %{y:.4f}<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Kepler Light Curve - {data['planet_info']['type']}",
                xaxis_title="Time (hours)",
                yaxis_title="Relative Flux",
                template="plotly_dark",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
        
        else:
            # Create Matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Main light curve
            ax.plot(time, flux, color='yellow', linewidth=1, label='Stellar Flux')
            
            # Highlight transits
            if analysis_results.get('transits'):
                transit_times = [transit['time'] for transit in analysis_results['transits']]
                transit_fluxes = [flux[np.argmin(np.abs(time - t))] for t in transit_times]
                ax.scatter(transit_times, transit_fluxes, color='red', s=80, marker='d', 
                          label='Detected Transits', zorder=5)
            
            # Highlight anomalies
            if analysis_results.get('anomalies'):
                anomaly_times = [anomaly['time'] for anomaly in analysis_results['anomalies']]
                anomaly_fluxes = [anomaly['flux'] for anomaly in analysis_results['anomalies']]
                ax.scatter(anomaly_times, anomaly_fluxes, color='orange', s=40, 
                          label='Anomalies', zorder=5)
            
            ax.set_title(f"Kepler Light Curve - {data['planet_info']['type']}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Relative Flux")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_astronomical_image(self, data: Dict, analysis_results: Dict, use_plotly: bool = True):
        """
        Create interactive plot for astronomical image
        
        Args:
            data (dict): Image data
            analysis_results (dict): Analysis results
            use_plotly (bool): Whether to use Plotly (True) or Matplotlib (False)
            
        Returns:
            plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        image = data["image"]
        
        if use_plotly:
            # Create Plotly figure
            fig = go.Figure()
            
            # Main image
            fig.add_trace(go.Heatmap(
                z=image,
                colorscale='Viridis',
                name='Astronomical Image',
                hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
            ))
            
            # Highlight detected objects
            if analysis_results.get('objects'):
                for obj in analysis_results['objects']:
                    center_x, center_y = obj['center']
                    fig.add_trace(go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode='markers',
                        name=f"Object {obj['id']}",
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='diamond',
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate=f"Object {obj['id']}<br>Brightness: {obj['brightness']:.3f}<br>Area: {obj['area']}<extra></extra>",
                        showlegend=False
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"Astronomical Image - {data['object_info']['type']}",
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)",
                template="plotly_dark",
                height=500,
                showlegend=False
            )
            
            return fig
        
        else:
            # Create Matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Main image
            im = ax.imshow(image, cmap='viridis', origin='lower')
            
            # Highlight detected objects
            if analysis_results.get('objects'):
                for obj in analysis_results['objects']:
                    center_x, center_y = obj['center']
                    ax.scatter(center_x, center_y, color='red', s=100, marker='d', 
                              edgecolors='white', linewidth=2, zorder=5)
                    ax.annotate(f"{obj['id']}", (center_x, center_y), 
                               xytext=(5, 5), textcoords='offset points', 
                               color='white', fontsize=8)
            
            ax.set_title(f"Astronomical Image - {data['object_info']['type']}")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Intensity')
            
            return fig
    
    def create_analysis_dashboard(self, data: Dict, analysis_results: Dict, data_type: str):
        """
        Create a comprehensive analysis dashboard
        
        Args:
            data (dict): Original data
            analysis_results (dict): Analysis results
            data_type (str): Type of data ('ligo', 'kepler', 'image')
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        if data_type == "ligo":
            return self._create_ligo_dashboard(data, analysis_results)
        elif data_type == "kepler":
            return self._create_kepler_dashboard(data, analysis_results)
        elif data_type == "image":
            return self._create_image_dashboard(data, analysis_results)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _create_ligo_dashboard(self, data: Dict, analysis_results: Dict):
        """Create LIGO analysis dashboard"""
        time = data["time"]
        signal = data["signal"]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Gravitational Wave Signal', 'Signal Statistics', 
                          'Peak Distribution', 'Anomaly Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Main signal plot
        fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Signal',
                                line=dict(color='cyan', width=1)), row=1, col=1)
        
        # Highlight peaks
        if analysis_results.get('peaks'):
            peak_times = [peak['time'] for peak in analysis_results['peaks']]
            peak_amplitudes = [peak['amplitude'] for peak in analysis_results['peaks']]
            fig.add_trace(go.Scatter(x=peak_times, y=peak_amplitudes, mode='markers',
                                   name='Peaks', marker=dict(color='red', size=8)), row=1, col=1)
        
        # Statistics histogram
        fig.add_trace(go.Histogram(x=signal, nbinsx=30, name='Signal Distribution',
                                  marker_color='blue', opacity=0.7), row=1, col=2)
        
        # Peak amplitude vs time
        if analysis_results.get('peaks'):
            peak_times = [peak['time'] for peak in analysis_results['peaks']]
            peak_amplitudes = [peak['amplitude'] for peak in analysis_results['peaks']]
            fig.add_trace(go.Scatter(x=peak_times, y=peak_amplitudes, mode='markers+lines',
                                   name='Peak Evolution', marker=dict(color='red', size=8)), row=2, col=1)
        
        # Anomaly analysis
        if analysis_results.get('anomalies'):
            anomaly_times = [anomaly['time'] for anomaly in analysis_results['anomalies']]
            anomaly_amplitudes = [anomaly['amplitude'] for anomaly in analysis_results['anomalies']]
            fig.add_trace(go.Scatter(x=anomaly_times, y=anomaly_amplitudes, mode='markers',
                                   name='Anomalies', marker=dict(color='orange', size=6)), row=2, col=2)
        
        fig.update_layout(
            title=f"LIGO Analysis Dashboard - {data['event_info']['type']}",
            template="plotly_dark",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_kepler_dashboard(self, data: Dict, analysis_results: Dict):
        """Create Kepler analysis dashboard"""
        time = data["time"]
        flux = data["flux"]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Light Curve', 'Flux Distribution', 
                          'Transit Analysis', 'Stellar Variability'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Main light curve
        fig.add_trace(go.Scatter(x=time, y=flux, mode='lines', name='Flux',
                                line=dict(color='yellow', width=1)), row=1, col=1)
        
        # Highlight transits
        if analysis_results.get('transits'):
            transit_times = [transit['time'] for transit in analysis_results['transits']]
            transit_fluxes = [flux[np.argmin(np.abs(time - t))] for t in transit_times]
            fig.add_trace(go.Scatter(x=transit_times, y=transit_fluxes, mode='markers',
                                   name='Transits', marker=dict(color='red', size=10)), row=1, col=1)
        
        # Flux distribution
        fig.add_trace(go.Histogram(x=flux, nbinsx=30, name='Flux Distribution',
                                  marker_color='orange', opacity=0.7), row=1, col=2)
        
        # Transit depth analysis
        if analysis_results.get('transits'):
            transit_depths = [transit['depth_percent'] for transit in analysis_results['transits']]
            transit_times = [transit['time'] for transit in analysis_results['transits']]
            fig.add_trace(go.Scatter(x=transit_times, y=transit_depths, mode='markers+lines',
                                   name='Transit Depths', marker=dict(color='red', size=8)), row=2, col=1)
        
        # Power spectrum for stellar variability
        if len(flux) > 100:
            freqs = np.fft.fftfreq(len(flux), d=time[1]-time[0])
            power = np.abs(np.fft.fft(flux))**2
            # Only positive frequencies
            pos_freqs = freqs[freqs > 0]
            pos_power = power[freqs > 0]
            fig.add_trace(go.Scatter(x=pos_freqs, y=pos_power, mode='lines',
                                   name='Power Spectrum', line=dict(color='green')), row=2, col=2)
        
        fig.update_layout(
            title=f"Kepler Analysis Dashboard - {data['planet_info']['type']}",
            template="plotly_dark",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_image_dashboard(self, data: Dict, analysis_results: Dict):
        """Create image analysis dashboard"""
        image = data["image"]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Astronomical Image', 'Intensity Distribution', 
                          'Object Detection', 'Brightness Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Main image
        fig.add_trace(go.Heatmap(z=image, colorscale='Viridis', name='Image'), row=1, col=1)
        
        # Intensity distribution
        fig.add_trace(go.Histogram(x=image.flatten(), nbinsx=50, name='Intensity Distribution',
                                  marker_color='purple', opacity=0.7), row=1, col=2)
        
        # Object detection visualization
        if analysis_results.get('objects'):
            obj_x = [obj['center'][0] for obj in analysis_results['objects']]
            obj_y = [obj['center'][1] for obj in analysis_results['objects']]
            obj_brightness = [obj['brightness'] for obj in analysis_results['objects']]
            
            fig.add_trace(go.Scatter(x=obj_x, y=obj_y, mode='markers',
                                   name='Detected Objects',
                                   marker=dict(color=obj_brightness, colorscale='Viridis', size=8)), row=2, col=1)
        
        # Brightness vs area scatter
        if analysis_results.get('objects'):
            obj_brightness = [obj['brightness'] for obj in analysis_results['objects']]
            obj_areas = [obj['area'] for obj in analysis_results['objects']]
            fig.add_trace(go.Scatter(x=obj_areas, y=obj_brightness, mode='markers',
                                   name='Brightness vs Area',
                                   marker=dict(color='red', size=8)), row=2, col=2)
        
        fig.update_layout(
            title=f"Image Analysis Dashboard - {data['object_info']['type']}",
            template="plotly_dark",
            height=800,
            showlegend=True
        )
        
        return fig

def create_summary_metrics(data: Dict, analysis_results: Dict, data_type: str) -> Dict:
    """
    Create summary metrics for display
    
    Args:
        data (dict): Original data
        analysis_results (dict): Analysis results
        data_type (str): Type of data
        
    Returns:
        dict: Summary metrics
    """
    metrics = {}
    
    if data_type == "ligo":
        metrics = {
            "Signal Classification": analysis_results.get('classification', 'Unknown'),
            "Peaks Detected": len(analysis_results.get('peaks', [])),
            "Anomalies Found": len(analysis_results.get('anomalies', [])),
            "RMS Amplitude": f"{analysis_results.get('statistics', {}).get('rms', 0):.6f}",
            "Signal Range": f"{analysis_results.get('statistics', {}).get('min', 0):.6f} to {analysis_results.get('statistics', {}).get('max', 0):.6f}"
        }
    
    elif data_type == "kepler":
        metrics = {
            "Classification": analysis_results.get('classification', 'Unknown'),
            "Transits Detected": len(analysis_results.get('transits', [])),
            "Anomalies Found": len(analysis_results.get('anomalies', [])),
            "Stellar Variability": f"{analysis_results.get('statistics', {}).get('std', 0):.4f}",
            "Flux Range": f"{analysis_results.get('statistics', {}).get('min', 0):.4f} to {analysis_results.get('statistics', {}).get('max', 0):.4f}"
        }
    
    elif data_type == "image":
        metrics = {
            "Classification": analysis_results.get('classification', 'Unknown'),
            "Objects Detected": len(analysis_results.get('objects', [])),
            "Mean Intensity": f"{analysis_results.get('statistics', {}).get('mean', 0):.3f}",
            "Image Size": f"{analysis_results.get('statistics', {}).get('shape', (0, 0))}"
        }
    
    return metrics

if __name__ == "__main__":
    # Test the visualizer
    from data_generators import AstrophysicsDataGenerator
    from ml_analysis import analyze_data
    
    generator = AstrophysicsDataGenerator()
    visualizer = AstrophysicsVisualizer()
    
    # Test LIGO visualization
    ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
    ligo_results = analyze_data(ligo_data, "ligo")
    ligo_fig = visualizer.plot_ligo_signal(ligo_data, ligo_results)
    
    # Test Kepler visualization
    kepler_data = generator.generate_kepler_light_curve("earth_like")
    kepler_results = analyze_data(kepler_data, "kepler")
    kepler_fig = visualizer.plot_kepler_light_curve(kepler_data, kepler_results)
    
    print("Visualization tests completed!") 