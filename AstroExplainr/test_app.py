"""
Test Script for AstroExplainr
Verifies that all modules work correctly
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from data_generators import AstrophysicsDataGenerator, create_sample_datasets
        print("✅ data_generators imported successfully")
    except Exception as e:
        print(f"❌ Failed to import data_generators: {e}")
        return False
    
    try:
        from ml_analysis import analyze_data, AstrophysicsAnalyzer
        print("✅ ml_analysis imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ml_analysis: {e}")
        return False
    
    try:
        from llm_explainer import AstrophysicsExplainer, explain_analysis
        print("✅ llm_explainer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import llm_explainer: {e}")
        return False
    
    try:
        from visualization import AstrophysicsVisualizer, create_summary_metrics
        print("✅ visualization imported successfully")
    except Exception as e:
        print(f"❌ Failed to import visualization: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation functionality"""
    print("\n🔍 Testing data generation...")
    
    try:
        generator = AstrophysicsDataGenerator()
        
        # Test LIGO data generation
        ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
        assert "time" in ligo_data
        assert "signal" in ligo_data
        assert "event_info" in ligo_data
        print("✅ LIGO data generation successful")
        
        # Test Kepler data generation
        kepler_data = generator.generate_kepler_light_curve("earth_like")
        assert "time" in kepler_data
        assert "flux" in kepler_data
        assert "planet_info" in kepler_data
        print("✅ Kepler data generation successful")
        
        # Test image generation
        image_data = generator.generate_astronomical_image("galaxy")
        assert "image" in image_data
        assert "object_info" in image_data
        print("✅ Image generation successful")
        
        # Test sample datasets
        datasets = create_sample_datasets()
        assert len(datasets) > 0
        print("✅ Sample datasets creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        traceback.print_exc()
        return False

def test_ml_analysis():
    """Test ML analysis functionality"""
    print("\n🔍 Testing ML analysis...")
    
    try:
        from data_generators import AstrophysicsDataGenerator
        
        generator = AstrophysicsDataGenerator()
        analyzer = AstrophysicsAnalyzer()
        
        # Test LIGO analysis
        ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
        ligo_results = analyze_data(ligo_data, "ligo")
        assert "classification" in ligo_results
        assert "peaks" in ligo_results
        assert "anomalies" in ligo_results
        print("✅ LIGO analysis successful")
        
        # Test Kepler analysis
        kepler_data = generator.generate_kepler_light_curve("earth_like")
        kepler_results = analyze_data(kepler_data, "kepler")
        assert "classification" in kepler_results
        assert "transits" in kepler_results
        assert "anomalies" in kepler_results
        print("✅ Kepler analysis successful")
        
        # Test image analysis
        image_data = generator.generate_astronomical_image("galaxy")
        image_results = analyze_data(image_data, "image")
        assert "classification" in image_results
        assert "objects" in image_results
        print("✅ Image analysis successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ML analysis failed: {e}")
        traceback.print_exc()
        return False

def test_llm_explainer():
    """Test LLM explainer functionality"""
    print("\n🔍 Testing LLM explainer...")
    
    try:
        from data_generators import AstrophysicsDataGenerator
        from ml_analysis import analyze_data
        
        generator = AstrophysicsDataGenerator()
        explainer = AstrophysicsExplainer()
        
        # Test LIGO explanation
        ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
        ligo_results = analyze_data(ligo_data, "ligo")
        ligo_explanation = explainer.explain_ligo_analysis(ligo_data, ligo_results)
        assert len(ligo_explanation) > 0
        print("✅ LIGO explanation successful")
        
        # Test Kepler explanation
        kepler_data = generator.generate_kepler_light_curve("earth_like")
        kepler_results = analyze_data(kepler_data, "kepler")
        kepler_explanation = explainer.explain_kepler_analysis(kepler_data, kepler_results)
        assert len(kepler_explanation) > 0
        print("✅ Kepler explanation successful")
        
        # Test image explanation
        image_data = generator.generate_astronomical_image("galaxy")
        image_results = analyze_data(image_data, "image")
        image_explanation = explainer.explain_image_analysis(image_data, image_results)
        assert len(image_explanation) > 0
        print("✅ Image explanation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM explainer failed: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\n🔍 Testing visualization...")
    
    try:
        from data_generators import AstrophysicsDataGenerator
        from ml_analysis import analyze_data
        
        generator = AstrophysicsDataGenerator()
        visualizer = AstrophysicsVisualizer()
        
        # Test LIGO visualization
        ligo_data = generator.generate_ligo_style_signal("binary_black_hole")
        ligo_results = analyze_data(ligo_data, "ligo")
        ligo_fig = visualizer.plot_ligo_signal(ligo_data, ligo_results)
        assert ligo_fig is not None
        print("✅ LIGO visualization successful")
        
        # Test Kepler visualization
        kepler_data = generator.generate_kepler_light_curve("earth_like")
        kepler_results = analyze_data(kepler_data, "kepler")
        kepler_fig = visualizer.plot_kepler_light_curve(kepler_data, kepler_results)
        assert kepler_fig is not None
        print("✅ Kepler visualization successful")
        
        # Test image visualization
        image_data = generator.generate_astronomical_image("galaxy")
        image_results = analyze_data(image_data, "image")
        image_fig = visualizer.plot_astronomical_image(image_data, image_results)
        assert image_fig is not None
        print("✅ Image visualization successful")
        
        # Test metrics creation
        ligo_metrics = create_summary_metrics(ligo_data, ligo_results, "ligo")
        assert len(ligo_metrics) > 0
        print("✅ Metrics creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration"""
    print("\n🔍 Testing full integration...")
    
    try:
        from data_generators import create_sample_datasets
        from ml_analysis import analyze_data
        from llm_explainer import AstrophysicsExplainer
        from visualization import AstrophysicsVisualizer
        
        # Get sample datasets
        datasets = create_sample_datasets()
        
        # Test each dataset type
        for dataset_name, data in datasets.items():
            if "ligo" in dataset_name:
                data_type = "ligo"
            elif "kepler" in dataset_name:
                data_type = "kepler"
            else:
                data_type = "image"
            
            # Run analysis
            results = analyze_data(data, data_type)
            
            # Generate explanation
            explainer = AstrophysicsExplainer()
            if data_type == "ligo":
                explanation = explainer.explain_ligo_analysis(data, results)
            elif data_type == "kepler":
                explanation = explainer.explain_kepler_analysis(data, results)
            else:
                explanation = explainer.explain_image_analysis(data, results)
            
            # Create visualization
            visualizer = AstrophysicsVisualizer()
            if data_type == "ligo":
                fig = visualizer.plot_ligo_signal(data, results)
            elif data_type == "kepler":
                fig = visualizer.plot_kepler_light_curve(data, results)
            else:
                fig = visualizer.plot_astronomical_image(data, results)
            
            print(f"✅ {dataset_name} integration test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting AstroExplainr Tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("ML Analysis", test_ml_analysis),
        ("LLM Explainer", test_llm_explainer),
        ("Visualization", test_visualization),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED\n")
            else:
                print(f"❌ {test_name} test FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}\n")
            traceback.print_exc()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AstroExplainr is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 