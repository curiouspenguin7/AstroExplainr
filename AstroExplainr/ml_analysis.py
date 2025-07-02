"""
ML Analysis Module for AstroExplainr
Performs machine learning analysis on astrophysics data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import norm
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class AstrophysicsAnalyzer:
    """Analyzes astrophysics data using various ML techniques"""
    
    def __init__(self):
        """Initialize the analyzer with ML models"""
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        
    def analyze_ligo_signal(self, time, signal_data):
        """
        Analyze LIGO-style gravitational wave signal
        
        Args:
            time (array): Time array
            signal_data (array): Signal data
            
        Returns:
            dict: Analysis results
        """
        results = {
            "anomalies": [],
            "peaks": [],
            "patterns": [],
            "statistics": {},
            "classification": "unknown"
        }
        
        # Basic statistics
        results["statistics"] = {
            "mean": np.mean(signal_data),
            "std": np.std(signal_data),
            "max": np.max(signal_data),
            "min": np.min(signal_data),
            "rms": np.sqrt(np.mean(signal_data**2))
        }
        
        # Peak detection
        peaks, properties = find_peaks(signal_data, height=np.std(signal_data), 
                                     distance=len(signal_data)//20)
        
        for i, peak_idx in enumerate(peaks):
            peak_info = {
                "index": int(peak_idx),
                "time": float(time[peak_idx]),
                "amplitude": float(signal_data[peak_idx]),
                "prominence": float(properties["prominences"][i]) if "prominences" in properties else 0
            }
            results["peaks"].append(peak_info)
        
        # Anomaly detection
        signal_reshaped = signal_data.reshape(-1, 1)
        anomaly_scores = self.isolation_forest.fit_predict(signal_reshaped)
        anomaly_indices = np.where(anomaly_scores == -1)[0]
        
        for idx in anomaly_indices:
            anomaly_info = {
                "index": int(idx),
                "time": float(time[idx]),
                "amplitude": float(signal_data[idx]),
                "score": float(anomaly_scores[idx])
            }
            results["anomalies"].append(anomaly_info)
        
        # Pattern analysis - look for chirp-like patterns
        # Calculate frequency evolution
        if len(signal_data) > 100:
            # Use spectrogram to detect frequency evolution
            f, t, Sxx = signal.spectrogram(signal_data, fs=1/(time[1]-time[0]), 
                                         nperseg=min(256, len(signal_data)//4))
            
            # Look for increasing frequency (chirp pattern)
            peak_freqs = f[np.argmax(Sxx, axis=0)]
            if len(peak_freqs) > 5:
                freq_trend = np.polyfit(range(len(peak_freqs)), peak_freqs, 1)[0]
                
                if freq_trend > 0:
                    results["patterns"].append({
                        "type": "chirp",
                        "description": "Increasing frequency pattern detected",
                        "confidence": min(abs(freq_trend) * 10, 0.95)
                    })
        
        # Classification based on features
        if len(results["peaks"]) > 0 and len(results["anomalies"]) > 0:
            max_peak = max(results["peaks"], key=lambda x: x["amplitude"])
            if max_peak["amplitude"] > 2 * np.std(signal_data):
                results["classification"] = "likely_merger"
            else:
                results["classification"] = "possible_signal"
        elif len(results["anomalies"]) > 0:
            results["classification"] = "noise_or_glitch"
        else:
            results["classification"] = "quiet_period"
        
        return results
    
    def analyze_kepler_light_curve(self, time, flux):
        """
        Analyze Kepler-style light curve for exoplanet transits
        
        Args:
            time (array): Time array
            flux (array): Flux data
            
        Returns:
            dict: Analysis results
        """
        results = {
            "transits": [],
            "anomalies": [],
            "variability": {},
            "statistics": {},
            "classification": "unknown"
        }
        
        # Basic statistics
        results["statistics"] = {
            "mean": np.mean(flux),
            "std": np.std(flux),
            "max": np.max(flux),
            "min": np.min(flux),
            "range": np.max(flux) - np.min(flux)
        }
        
        # Detect transits (dips in flux)
        # Use rolling median to detrend
        window = min(50, len(flux)//10)
        if window > 1:
            detrended = flux - pd.Series(flux).rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')
        else:
            detrended = flux - np.mean(flux)
        
        # Find dips (transits)
        dips, properties = find_peaks(-detrended, height=np.std(detrended), 
                                    distance=len(flux)//10)
        
        for i, dip_idx in enumerate(dips):
            # Calculate transit properties
            dip_depth = -detrended[dip_idx]
            dip_width = self._estimate_transit_width(detrended, dip_idx)
            
            transit_info = {
                "index": int(dip_idx),
                "time": float(time[dip_idx]),
                "depth": float(dip_depth),
                "width": float(dip_width),
                "depth_percent": float(dip_depth / np.mean(flux) * 100)
            }
            results["transits"].append(transit_info)
        
        # Anomaly detection
        flux_reshaped = flux.reshape(-1, 1)
        anomaly_scores = self.isolation_forest.fit_predict(flux_reshaped)
        anomaly_indices = np.where(anomaly_scores == -1)[0]
        
        for idx in anomaly_indices:
            anomaly_info = {
                "index": int(idx),
                "time": float(time[idx]),
                "flux": float(flux[idx]),
                "deviation": float((flux[idx] - np.mean(flux)) / np.std(flux))
            }
            results["anomalies"].append(anomaly_info)
        
        # Stellar variability analysis
        # Calculate power spectrum
        if len(flux) > 100:
            freqs = np.fft.fftfreq(len(flux), d=time[1]-time[0])
            power = np.abs(np.fft.fft(flux))**2
            
            # Find dominant frequencies
            peak_freqs = freqs[np.argsort(power)[-5:]]
            results["variability"] = {
                "dominant_frequencies": peak_freqs.tolist(),
                "total_variability": float(np.std(flux) / np.mean(flux))
            }
        
        # Classification
        if len(results["transits"]) > 0:
            max_transit = max(results["transits"], key=lambda x: x["depth"])
            if max_transit["depth_percent"] > 0.1:  # >0.1% transit depth
                results["classification"] = "exoplanet_candidate"
            else:
                results["classification"] = "possible_transit"
        elif results["statistics"]["std"] > 0.01:
            results["classification"] = "variable_star"
        else:
            results["classification"] = "quiet_star"
        
        return results
    
    def analyze_astronomical_image(self, image):
        """
        Analyze astronomical image for objects and features
        
        Args:
            image (array): 2D image array
            
        Returns:
            dict: Analysis results
        """
        results = {
            "objects": [],
            "statistics": {},
            "classification": "unknown"
        }
        
        # Basic statistics
        results["statistics"] = {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "max": float(np.max(image)),
            "min": float(np.min(image)),
            "shape": image.shape
        }
        
        # Object detection using thresholding
        threshold = np.mean(image) + 2 * np.std(image)
        bright_pixels = image > threshold
        
        if np.sum(bright_pixels) > 0:
            # Find connected components
            from scipy import ndimage
            labeled, num_features = ndimage.label(bright_pixels)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > 5:  # Minimum size threshold
                    # Calculate object properties
                    coords = np.where(component)
                    center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
                    brightness = np.mean(image[component])
                    area = np.sum(component)
                    
                    object_info = {
                        "id": i,
                        "center": [float(center_x), float(center_y)],
                        "brightness": float(brightness),
                        "area": int(area),
                        "type": "bright_object"
                    }
                    results["objects"].append(object_info)
        
        # Classification based on object count and distribution
        if len(results["objects"]) > 10:
            results["classification"] = "star_field"
        elif len(results["objects"]) > 1:
            results["classification"] = "multiple_objects"
        elif len(results["objects"]) == 1:
            results["classification"] = "single_object"
        else:
            results["classification"] = "empty_field"
        
        return results
    
    def _estimate_transit_width(self, detrended, peak_idx):
        """Estimate the width of a transit dip"""
        # Find where the dip crosses zero
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and detrended[left_idx] < 0:
            left_idx -= 1
        
        while right_idx < len(detrended) - 1 and detrended[right_idx] < 0:
            right_idx += 1
        
        return right_idx - left_idx
    
    def generate_analysis_summary(self, data_type, results):
        """
        Generate a human-readable summary of analysis results
        
        Args:
            data_type (str): Type of data analyzed
            results (dict): Analysis results
            
        Returns:
            str: Summary text
        """
        if data_type == "ligo":
            summary = f"Analysis of gravitational wave signal:\n"
            summary += f"- Signal classification: {results['classification']}\n"
            summary += f"- Detected {len(results['peaks'])} significant peaks\n"
            summary += f"- Found {len(results['anomalies'])} anomalies\n"
            summary += f"- RMS amplitude: {results['statistics']['rms']:.4f}\n"
            
            if results['peaks']:
                max_peak = max(results['peaks'], key=lambda x: x['amplitude'])
                summary += f"- Strongest peak at t={max_peak['time']:.2f}s with amplitude {max_peak['amplitude']:.4f}\n"
            
            if results['patterns']:
                for pattern in results['patterns']:
                    summary += f"- Pattern detected: {pattern['description']} (confidence: {pattern['confidence']:.2f})\n"
        
        elif data_type == "kepler":
            summary = f"Analysis of light curve:\n"
            summary += f"- Classification: {results['classification']}\n"
            summary += f"- Detected {len(results['transits'])} potential transits\n"
            summary += f"- Found {len(results['anomalies'])} anomalies\n"
            summary += f"- Stellar variability: {results['statistics']['std']:.4f}\n"
            
            if results['transits']:
                max_transit = max(results['transits'], key=lambda x: x['depth'])
                summary += f"- Deepest transit: {max_transit['depth_percent']:.3f}% at t={max_transit['time']:.2f}h\n"
        
        elif data_type == "image":
            summary = f"Analysis of astronomical image:\n"
            summary += f"- Classification: {results['classification']}\n"
            summary += f"- Detected {len(results['objects'])} objects\n"
            summary += f"- Image statistics: mean={results['statistics']['mean']:.3f}, std={results['statistics']['std']:.3f}\n"
            
            if results['objects']:
                brightest = max(results['objects'], key=lambda x: x['brightness'])
                summary += f"- Brightest object at position {brightest['center']} with brightness {brightest['brightness']:.3f}\n"
        
        return summary

def analyze_data(data, data_type):
    """
    Main function to analyze astrophysics data
    
    Args:
        data (dict): Data dictionary with time/signal/flux/image
        data_type (str): Type of data ('ligo', 'kepler', 'image')
        
    Returns:
        dict: Analysis results
    """
    analyzer = AstrophysicsAnalyzer()
    
    if data_type == "ligo":
        return analyzer.analyze_ligo_signal(data["time"], data["signal"])
    elif data_type == "kepler":
        return analyzer.analyze_kepler_light_curve(data["time"], data["flux"])
    elif data_type == "image":
        return analyzer.analyze_astronomical_image(data["image"])
    else:
        raise ValueError(f"Unknown data type: {data_type}")

if __name__ == "__main__":
    # Test the analyzer
    from data_generators import AstrophysicsDataGenerator
    
    generator = AstrophysicsDataGenerator()
    analyzer = AstrophysicsAnalyzer()
    
    # Test LIGO analysis
    ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
    ligo_results = analyzer.analyze_ligo_signal(ligo_data["time"], ligo_data["signal"])
    print("LIGO Analysis Results:")
    print(analyzer.generate_analysis_summary("ligo", ligo_results))
    
    # Test Kepler analysis
    kepler_data = generator.generate_kepler_light_curve("earth_like")
    kepler_results = analyzer.analyze_kepler_light_curve(kepler_data["time"], kepler_data["flux"])
    print("\nKepler Analysis Results:")
    print(analyzer.generate_analysis_summary("kepler", kepler_results)) 