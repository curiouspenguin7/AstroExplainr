"""
LLM Explainer Module for AstroExplainr
Uses large language models to generate explanations of astrophysics data analysis results.
"""

import os
import json
from typing import Dict, List, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AstrophysicsExplainer:
    """Uses LLMs to explain astrophysics data analysis results"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the explainer
        
        Args:
            api_key (str): OpenAI API key (if None, uses environment variable)
            model (str): OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("Warning: No OpenAI API key found. Explanations will be simulated.")
    
    def explain_ligo_analysis(self, data: Dict, analysis_results: Dict) -> str:
        """
        Generate explanation for LIGO gravitational wave analysis
        
        Args:
            data (dict): Original LIGO data
            analysis_results (dict): ML analysis results
            
        Returns:
            str: Natural language explanation
        """
        if not self.api_key:
            return self._simulate_ligo_explanation(data, analysis_results)
        
        # Prepare context for the LLM
        context = self._prepare_ligo_context(data, analysis_results)
        
        prompt = f"""
You are an expert astrophysicist specializing in gravitational wave astronomy. 
Analyze the following LIGO data and provide a clear, educational explanation of what was detected.

Data Context:
{context}

Please provide an explanation that:
1. Describes what was detected in the signal
2. Explains the significance of the findings
3. Uses accessible language while maintaining scientific accuracy
4. Mentions any uncertainties or limitations
5. Suggests what this might mean for our understanding of the universe

Keep the explanation under 300 words and suitable for both scientists and interested laypeople.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert astrophysicist and science communicator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._simulate_ligo_explanation(data, analysis_results)
    
    def explain_kepler_analysis(self, data: Dict, analysis_results: Dict) -> str:
        """
        Generate explanation for Kepler exoplanet analysis
        
        Args:
            data (dict): Original Kepler data
            analysis_results (dict): ML analysis results
            
        Returns:
            str: Natural language explanation
        """
        if not self.api_key:
            return self._simulate_kepler_explanation(data, analysis_results)
        
        # Prepare context for the LLM
        context = self._prepare_kepler_context(data, analysis_results)
        
        prompt = f"""
You are an expert exoplanet astronomer analyzing Kepler space telescope data.
Analyze the following light curve data and provide a clear explanation of what was detected.

Data Context:
{context}

Please provide an explanation that:
1. Describes what was detected in the light curve
2. Explains the significance for exoplanet discovery
3. Uses accessible language while maintaining scientific accuracy
4. Mentions any uncertainties or alternative explanations
5. Suggests what follow-up observations might be needed

Keep the explanation under 300 words and suitable for both scientists and interested laypeople.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert exoplanet astronomer and science communicator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._simulate_kepler_explanation(data, analysis_results)
    
    def explain_image_analysis(self, data: Dict, analysis_results: Dict) -> str:
        """
        Generate explanation for astronomical image analysis
        
        Args:
            data (dict): Original image data
            analysis_results (dict): ML analysis results
            
        Returns:
            str: Natural language explanation
        """
        if not self.api_key:
            return self._simulate_image_explanation(data, analysis_results)
        
        # Prepare context for the LLM
        context = self._prepare_image_context(data, analysis_results)
        
        prompt = f"""
You are an expert observational astronomer analyzing astronomical images.
Analyze the following image data and provide a clear explanation of what was detected.

Data Context:
{context}

Please provide an explanation that:
1. Describes what objects or features were detected
2. Explains the significance of the findings
3. Uses accessible language while maintaining scientific accuracy
4. Mentions any uncertainties or limitations
5. Suggests what this tells us about the astronomical object

Keep the explanation under 300 words and suitable for both scientists and interested laypeople.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert observational astronomer and science communicator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._simulate_image_explanation(data, analysis_results)
    
    def _prepare_ligo_context(self, data: Dict, analysis_results: Dict) -> str:
        """Prepare context string for LIGO analysis"""
        context = f"""
Signal Duration: {data['metadata']['duration']} seconds
Sample Rate: {data['metadata']['sample_rate']} Hz
Event Type: {data['event_info']['type']}

Analysis Results:
- Classification: {analysis_results['classification']}
- Number of peaks detected: {len(analysis_results['peaks'])}
- Number of anomalies: {len(analysis_results['anomalies'])}
- RMS amplitude: {analysis_results['statistics']['rms']:.4f}
- Signal range: {analysis_results['statistics']['min']:.4f} to {analysis_results['statistics']['max']:.4f}

Peak Details:
"""
        if analysis_results['peaks']:
            for i, peak in enumerate(analysis_results['peaks'][:3]):  # Top 3 peaks
                context += f"- Peak {i+1}: Time={peak['time']:.2f}s, Amplitude={peak['amplitude']:.4f}\n"
        
        if analysis_results['patterns']:
            context += "\nPatterns Detected:\n"
            for pattern in analysis_results['patterns']:
                context += f"- {pattern['description']} (confidence: {pattern['confidence']:.2f})\n"
        
        return context
    
    def _prepare_kepler_context(self, data: Dict, analysis_results: Dict) -> str:
        """Prepare context string for Kepler analysis"""
        context = f"""
Light Curve Duration: {data['metadata']['duration']} hours
Sample Rate: {data['metadata']['sample_rate']} Hz
Planet Type: {data['planet_info']['type']}

Analysis Results:
- Classification: {analysis_results['classification']}
- Number of transits detected: {len(analysis_results['transits'])}
- Number of anomalies: {len(analysis_results['anomalies'])}
- Stellar variability: {analysis_results['statistics']['std']:.4f}
- Flux range: {analysis_results['statistics']['min']:.4f} to {analysis_results['statistics']['max']:.4f}

Transit Details:
"""
        if analysis_results['transits']:
            for i, transit in enumerate(analysis_results['transits'][:3]):  # Top 3 transits
                context += f"- Transit {i+1}: Time={transit['time']:.2f}h, Depth={transit['depth_percent']:.3f}%\n"
        
        if 'variability' in analysis_results:
            context += f"\nStellar Variability: {analysis_results['variability']['total_variability']:.4f}\n"
        
        return context
    
    def _prepare_image_context(self, data: Dict, analysis_results: Dict) -> str:
        """Prepare context string for image analysis"""
        context = f"""
Image Size: {analysis_results['statistics']['shape']}
Object Type: {data['object_info']['type']}

Analysis Results:
- Classification: {analysis_results['classification']}
- Number of objects detected: {len(analysis_results['objects'])}
- Image statistics: mean={analysis_results['statistics']['mean']:.3f}, std={analysis_results['statistics']['std']:.3f}

Object Details:
"""
        if analysis_results['objects']:
            for i, obj in enumerate(analysis_results['objects'][:5]):  # Top 5 objects
                context += f"- Object {i+1}: Position={obj['center']}, Brightness={obj['brightness']:.3f}, Area={obj['area']}\n"
        
        return context
    
    def _simulate_ligo_explanation(self, data: Dict, analysis_results: Dict) -> str:
        """Simulate LIGO explanation when API is not available"""
        classification = analysis_results['classification']
        
        if classification == "likely_merger":
            return """This gravitational wave signal shows strong evidence of a binary black hole merger. 
            The detected peaks and increasing frequency pattern (chirp) are characteristic of two massive 
            objects spiraling inward and merging. The signal amplitude and duration suggest this was a 
            significant event, possibly involving black holes with masses of several tens of solar masses. 
            Such detections help us understand the population of binary black holes in the universe and 
            test Einstein's theory of general relativity in extreme gravitational fields."""
        
        elif classification == "possible_signal":
            return """The analysis detected some interesting features in this gravitational wave data, 
            including peaks and anomalies that could indicate a real astrophysical signal. However, 
            the signal strength is relatively weak, making it difficult to definitively classify. 
            This could be a distant merger event, instrumental noise, or a combination of both. 
            Follow-up analysis with additional detectors or longer observation times would help 
            confirm the nature of this signal."""
        
        else:
            return """This gravitational wave data appears to be dominated by noise or instrumental 
            glitches. While some anomalies were detected, they lack the characteristic patterns 
            expected from real astrophysical sources like binary mergers. This is common in 
            gravitational wave astronomy, where the vast majority of data contains only noise. 
            Such quiet periods are important for understanding detector performance and background 
            noise characteristics."""
    
    def _simulate_kepler_explanation(self, data: Dict, analysis_results: Dict) -> str:
        """Simulate Kepler explanation when API is not available"""
        classification = analysis_results['classification']
        
        if classification == "exoplanet_candidate":
            return """This light curve shows clear evidence of an exoplanet transit! The periodic 
            dips in stellar brightness indicate that a planet is passing in front of its host star, 
            blocking a small fraction of the star's light. The transit depth suggests this is likely 
            a substantial planet, possibly a gas giant or super-Earth. The shape and duration of 
            the transit provide clues about the planet's size and orbital characteristics. This is 
            exactly the type of signal that has led to the discovery of thousands of exoplanets 
            by the Kepler mission."""
        
        elif classification == "possible_transit":
            return """The light curve analysis detected subtle dips that could indicate a planetary 
            transit, though the signal is relatively weak. This might be a small planet (like Earth) 
            transiting a large star, or the transit might be grazing (the planet only partially 
            covers the star). The stellar variability present could also be masking the transit 
            signal. Additional observations or longer time series data would help confirm whether 
            this represents a real exoplanet or just stellar activity."""
        
        else:
            return """This light curve shows typical stellar variability without clear evidence 
            of planetary transits. The variations in brightness are likely due to stellar activity, 
            such as starspots, pulsations, or other intrinsic stellar processes. While no exoplanet 
            transits were detected, this data is still valuable for understanding stellar behavior 
            and establishing baseline measurements for future transit searches."""
    
    def _simulate_image_explanation(self, data: Dict, analysis_results: Dict) -> str:
        """Simulate image explanation when API is not available"""
        classification = analysis_results['classification']
        
        if classification == "star_field":
            return """This astronomical image shows a rich star field with numerous stellar objects 
            detected. The distribution and brightness of the stars suggest this could be part of 
            our Milky Way galaxy, possibly in the direction of a spiral arm or star-forming region. 
            The variety of stellar brightnesses indicates stars of different masses and evolutionary 
            stages. Such images are valuable for stellar population studies and understanding the 
            structure of our galaxy."""
        
        elif classification == "single_object":
            return """The image analysis detected a single bright object, which could be a star, 
            galaxy, or other astronomical source. The object's brightness and spatial extent provide 
            clues about its nature. This might be a foreground star in our galaxy, a distant galaxy, 
            or another type of astronomical object. Follow-up observations with different filters 
            or spectroscopic analysis would help determine the object's true nature and distance."""
        
        else:
            return """This astronomical image appears to show a relatively empty field with minimal 
            detectable objects. This could be a region of low stellar density, a field pointing 
            away from the galactic plane, or an area dominated by background noise. Such 'empty' 
            fields are important for establishing background levels and can sometimes reveal very 
            faint or distant objects that require deeper observations to detect."""
    
    def compare_explanations(self, user_explanation: str, ai_explanation: str) -> str:
        """
        Compare user and AI explanations
        
        Args:
            user_explanation (str): User's interpretation
            ai_explanation (str): AI-generated explanation
            
        Returns:
            str: Comparison analysis
        """
        if not self.api_key:
            return self._simulate_comparison(user_explanation, ai_explanation)
        
        prompt = f"""
Compare these two explanations of the same astrophysics data and provide insights:

User Explanation:
{user_explanation}

AI Explanation:
{ai_explanation}

Please provide a brief analysis that:
1. Identifies areas of agreement between the explanations
2. Notes any significant differences in interpretation
3. Suggests which aspects each explanation handles well
4. Provides constructive feedback for both perspectives

Keep the comparison under 200 words and be encouraging and educational in tone.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert science educator helping students understand astrophysics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._simulate_comparison(user_explanation, ai_explanation)
    
    def _simulate_comparison(self, user_explanation: str, ai_explanation: str) -> str:
        """Simulate explanation comparison when API is not available"""
        return """Both explanations provide valuable perspectives on the data! The AI explanation 
        offers a systematic analysis based on the detected features, while your interpretation 
        brings a unique human perspective. This kind of comparison is exactly how real scientific 
        collaboration works - combining automated analysis with human insight and intuition. 
        Consider how your observations might complement the AI's findings, and what additional 
        questions or hypotheses you might explore together."""

def explain_analysis(data: Dict, analysis_results: Dict, data_type: str) -> str:
    """
    Main function to generate explanations for astrophysics data
    
    Args:
        data (dict): Original data
        analysis_results (dict): ML analysis results
        data_type (str): Type of data ('ligo', 'kepler', 'image')
        
    Returns:
        str: Generated explanation
    """
    explainer = AstrophysicsExplainer()
    
    if data_type == "ligo":
        return explainer.explain_ligo_analysis(data, analysis_results)
    elif data_type == "kepler":
        return explainer.explain_kepler_analysis(data, analysis_results)
    elif data_type == "image":
        return explainer.explain_image_analysis(data, analysis_results)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

if __name__ == "__main__":
    # Test the explainer
    from data_generators import AstrophysicsDataGenerator
    from ml_analysis import analyze_data
    
    generator = AstrophysicsDataGenerator()
    explainer = AstrophysicsExplainer()
    
    # Test LIGO explanation
    ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
    ligo_results = analyze_data(ligo_data, "ligo")
    ligo_explanation = explainer.explain_ligo_analysis(ligo_data, ligo_results)
    print("LIGO Explanation:")
    print(ligo_explanation)
    
    # Test Kepler explanation
    kepler_data = generator.generate_kepler_light_curve("earth_like")
    kepler_results = analyze_data(kepler_data, "kepler")
    kepler_explanation = explainer.explain_kepler_analysis(kepler_data, kepler_results)
    print("\nKepler Explanation:")
    print(kepler_explanation) 