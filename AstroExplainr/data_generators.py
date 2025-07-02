"""
Data Generators for AstroExplainr
Generates simulated astrophysics data for testing and demonstration
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AstrophysicsDataGenerator:
    """Generates simulated astrophysics data for various phenomena"""
    
    def __init__(self, sample_rate=4096, duration=10):
        """
        Initialize the data generator
        
        Args:
            sample_rate (int): Sampling rate in Hz
            duration (float): Duration of data in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.time = np.linspace(0, duration, int(sample_rate * duration))
        
    def generate_ligo_style_signal(self, event_type="binary_black_hole", noise_level=0.1):
        """
        Generate LIGO-style gravitational wave signal
        
        Args:
            event_type (str): Type of event ('binary_black_hole', 'neutron_star', 'noise')
            noise_level (float): Level of noise to add
            
        Returns:
            dict: Signal data with metadata
        """
        # Generate noise baseline
        noise = np.random.normal(0, noise_level, len(self.time))
        
        if event_type == "binary_black_hole":
            # Simulate binary black hole merger signal
            # Chirp signal: frequency increases over time
            chirp_freq = 50 + 100 * (self.time / self.duration)**2
            chirp_amplitude = 0.5 * np.exp(-(self.time - self.duration/2)**2 / 0.5)
            signal_data = chirp_amplitude * np.sin(2 * np.pi * chirp_freq * self.time)
            
            # Add merger peak
            peak_time = self.duration * 0.6
            peak_width = 0.1
            peak = 2.0 * np.exp(-(self.time - peak_time)**2 / (2 * peak_width**2))
            signal_data += peak
            
            event_info = {
                "type": "Binary Black Hole Merger",
                "mass1": "30 M☉",
                "mass2": "25 M☉",
                "distance": "1.3 Gpc",
                "confidence": "99.9%"
            }
            
        elif event_type == "neutron_star":
            # Simulate neutron star merger signal
            freq = 1000
            amplitude = 0.3
            signal_data = amplitude * np.sin(2 * np.pi * freq * self.time)
            
            # Add burst at the end
            burst_time = self.duration * 0.8
            burst = 1.5 * np.exp(-(self.time - burst_time)**2 / 0.05)
            signal_data += burst
            
            event_info = {
                "type": "Neutron Star Merger",
                "mass1": "1.4 M☉",
                "mass2": "1.3 M☉",
                "distance": "130 Mpc",
                "confidence": "95.2%"
            }
            
        else:  # noise
            signal_data = np.zeros_like(self.time)
            event_info = {
                "type": "Noise/Glitch",
                "description": "Random noise or instrumental glitch",
                "confidence": "0%"
            }
        
        # Add noise
        signal_data += noise
        
        return {
            "time": self.time,
            "signal": signal_data,
            "event_info": event_info,
            "metadata": {
                "sample_rate": self.sample_rate,
                "duration": self.duration,
                "event_type": event_type,
                "noise_level": noise_level
            }
        }
    
    def generate_kepler_light_curve(self, planet_type="earth_like", noise_level=0.01):
        """
        Generate Kepler-style exoplanet transit light curve
        
        Args:
            planet_type (str): Type of planet ('earth_like', 'jupiter_like', 'no_transit')
            noise_level (float): Level of noise to add
            
        Returns:
            dict: Light curve data with metadata
        """
        # Generate baseline stellar flux
        baseline_flux = 1.0 + 0.01 * np.sin(2 * np.pi * 0.1 * self.time)  # Stellar variability
        
        if planet_type == "earth_like":
            # Earth-like planet transit
            transit_depth = 0.0084  # ~84 ppm for Earth around Sun-like star
            transit_duration = 0.5  # hours
            transit_center = self.duration * 0.5
            
            # Create transit dip
            transit = np.ones_like(self.time)
            transit_mask = (self.time >= transit_center - transit_duration/2) & \
                          (self.time <= transit_center + transit_duration/2)
            transit[transit_mask] = 1 - transit_depth
            
            flux = baseline_flux * transit
            
            planet_info = {
                "type": "Earth-like Exoplanet",
                "radius": "1.0 R⊕",
                "period": "365 days",
                "transit_depth": f"{transit_depth*1000:.1f} ppm",
                "confidence": "High"
            }
            
        elif planet_type == "jupiter_like":
            # Jupiter-like planet transit
            transit_depth = 0.01  # ~1% for Jupiter
            transit_duration = 0.3  # hours
            transit_center = self.duration * 0.5
            
            # Create transit dip
            transit = np.ones_like(self.time)
            transit_mask = (self.time >= transit_center - transit_duration/2) & \
                          (self.time <= transit_center + transit_duration/2)
            transit[transit_mask] = 1 - transit_depth
            
            flux = baseline_flux * transit
            
            planet_info = {
                "type": "Jupiter-like Exoplanet",
                "radius": "11.2 R⊕",
                "period": "12 years",
                "transit_depth": f"{transit_depth*100:.1f}%",
                "confidence": "High"
            }
            
        else:  # no transit
            flux = baseline_flux
            planet_info = {
                "type": "No Transit Detected",
                "description": "Stellar variability only",
                "confidence": "N/A"
            }
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(self.time))
        flux += noise
        
        return {
            "time": self.time,
            "flux": flux,
            "planet_info": planet_info,
            "metadata": {
                "sample_rate": self.sample_rate,
                "duration": self.duration,
                "planet_type": planet_type,
                "noise_level": noise_level
            }
        }
    
    def generate_astronomical_image(self, object_type="galaxy", size=(256, 256)):
        """
        Generate simulated astronomical image
        
        Args:
            object_type (str): Type of astronomical object
            size (tuple): Image dimensions
            
        Returns:
            dict: Image data with metadata
        """
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        
        if object_type == "galaxy":
            # Simulate spiral galaxy
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            # Spiral arms
            spiral = np.cos(3 * theta + 5 * r)
            bulge = np.exp(-r**2 / 0.3)
            
            image = 0.5 * bulge + 0.3 * spiral * np.exp(-r / 0.8)
            
            object_info = {
                "type": "Spiral Galaxy",
                "classification": "Sb",
                "redshift": "0.023",
                "magnitude": "14.2"
            }
            
        elif object_type == "star_cluster":
            # Simulate star cluster
            image = np.zeros(size)
            
            # Add random stars
            n_stars = 50
            for _ in range(n_stars):
                cx = np.random.randint(0, size[0])
                cy = np.random.randint(0, size[1])
                brightness = np.random.uniform(0.1, 1.0)
                
                # Gaussian star profile
                for i in range(max(0, cx-3), min(size[0], cx+4)):
                    for j in range(max(0, cy-3), min(size[1], cy+4)):
                        dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                        image[i, j] += brightness * np.exp(-dist**2 / 2)
            
            object_info = {
                "type": "Open Star Cluster",
                "age": "100 Myr",
                "distance": "2.5 kpc",
                "stars": "~100"
            }
            
        else:  # noise
            image = np.random.normal(0, 0.1, size)
            object_info = {
                "type": "Noise/Empty Field",
                "description": "Background noise or empty field",
                "magnitude": "N/A"
            }
        
        return {
            "image": image,
            "object_info": object_info,
            "metadata": {
                "size": size,
                "object_type": object_type
            }
        }

def create_sample_datasets():
    """Create a collection of sample datasets for the app"""
    generator = AstrophysicsDataGenerator()
    
    datasets = {
        "ligo_binary_black_hole": generator.generate_ligo_style_signal("binary_black_hole"),
        "ligo_neutron_star": generator.generate_ligo_style_signal("neutron_star"),
        "ligo_noise": generator.generate_ligo_style_signal("noise"),
        "kepler_earth_like": generator.generate_kepler_light_curve("earth_like"),
        "kepler_jupiter_like": generator.generate_kepler_light_curve("jupiter_like"),
        "kepler_no_transit": generator.generate_kepler_light_curve("no_transit"),
        "galaxy_image": generator.generate_astronomical_image("galaxy"),
        "star_cluster_image": generator.generate_astronomical_image("star_cluster")
    }
    
    return datasets

if __name__ == "__main__":
    # Test data generation
    generator = AstrophysicsDataGenerator()
    
    # Generate test signals
    ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
    kepler_data = generator.generate_kepler_light_curve("earth_like")
    
    # Plot test data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(ligo_data["time"], ligo_data["signal"])
    ax1.set_title("Simulated LIGO Signal (Binary Black Hole Merger)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Strain")
    
    ax2.plot(kepler_data["time"], kepler_data["flux"])
    ax2.set_title("Simulated Kepler Light Curve (Earth-like Planet)")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Relative Flux")
    
    plt.tight_layout()
    plt.show() 