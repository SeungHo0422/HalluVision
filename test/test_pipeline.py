"""
Pipeline testing and validation script
파이프라인 테스트 및 검증 스크립트
"""

import os
import json
import tempfile
from pathlib import Path

from ..main import HallucinationVisionPipeline
from ..config import Settings


def create_test_data():
    """테스트용 데이터 생성"""
    # Create temporary malware list
    test_malware_list = ["zeus", "stuxnet"]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for malware in test_malware_list:
            f.write(f"{malware}\n")
        malware_file_path = f.name
    
    # Create temporary original NER data
    original_ner_data = {
        "0": {
            "Malware": ["Zeus"],
            "System": ["Windows"],
            "Indicator": ["network traffic"],
            "Vulnerability": ["banking vulnerability"],
            "Organization": ["cybercriminal group"]
        },
        "1": {
            "Malware": ["Stuxnet"],
            "System": ["Windows", "SCADA"],
            "Indicator": ["USB spreading"],
            "Vulnerability": ["industrial control vulnerability"],
            "Organization": ["nation-state"]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(original_ner_data, f, indent=2)
        ner_file_path = f.name
    
    return malware_file_path, ner_file_path


def test_individual_components():
    """개별 컴포넌트 테스트"""
    print("=" * 50)
    print("Testing Individual Components")
    print("=" * 50)
    
    # Test Data Scraper
    print("1. Testing Data Scraper...")
    try:
        from .data_scraper import DataScraper
        scraper = DataScraper()
        
        # Test with sample malware list
        test_malware = ["zeus"]
        results = scraper.scrape_malware_data(test_malware, sources=["threatpost"])
        
        if results and len(results) > 0:
            print("   ✅ Data Scraper working")
        else:
            print("   ⚠️ Data Scraper returned empty results")
    except Exception as e:
        print(f"   ❌ Data Scraper failed: {e}")
    
    # Test LLM Proxy (only if API key available)
    if Settings.validate_api_key():
        print("2. Testing LLM Proxy...")
        try:
            from .llm_proxy import AbstractiveSummarizer, QuestionAnsweringHandler
            
            # Test Summarizer
            summarizer = AbstractiveSummarizer()
            test_text = "Zeus is a banking trojan that targets Windows systems."
            summary = summarizer.summarize_article(test_text)
            
            if summary and len(summary) > 0:
                print("   ✅ Summarizer working")
            else:
                print("   ⚠️ Summarizer returned empty result")
            
            # Test QA Handler
            qa_handler = QuestionAnsweringHandler()
            qa_result = qa_handler.qa_article(test_text)
            
            if qa_result and len(qa_result) > 0:
                print("   ✅ QA Handler working")
            else:
                print("   ⚠️ QA Handler returned empty result")
                
        except Exception as e:
            print(f"   ❌ LLM Proxy failed: {e}")
    else:
        print("2. Skipping LLM Proxy test (no API key)")
    
    # Test Preprocessors
    print("3. Testing Preprocessors...")
    try:
        from .preprocessor import SummaryPreprocessor, QAPreprocessor
        
        # Test Summary Preprocessor
        sum_preprocessor = SummaryPreprocessor()
        test_summary_results = [{"id": 0, "summary_text": "Test summary"}]
        prepared = sum_preprocessor.prepare_for_evaluation(test_summary_results)
        
        if prepared:
            print("   ✅ Summary Preprocessor working")
        
        # Test QA Preprocessor
        qa_preprocessor = QAPreprocessor()
        test_qa_results = [{"id": 0, "qa_text": "{Zeus, Windows, vulnerability, indicator, organization}"}]
        processed = qa_preprocessor.preprocess_qa_results(test_qa_results)
        
        if processed:
            print("   ✅ QA Preprocessor working")
            
    except Exception as e:
        print(f"   ❌ Preprocessors failed: {e}")
    
    # Test Analyzers
    print("4. Testing Analyzers...")
    try:
        from .analyzer import SummaryAnalyzer, QAAnalyzer, VisualizationManager
        
        # Test Summary Analyzer
        sum_analyzer = SummaryAnalyzer()
        test_original = {"0": {"Malware": ["Zeus"]}}
        test_summary = {"0": {"Malware": ["Zeus"]}}
        results = sum_analyzer.evaluate_model(test_original, test_summary)
        
        if results:
            print("   ✅ Summary Analyzer working")
        
        # Test QA Analyzer
        qa_analyzer = QAAnalyzer()
        test_qa_data = [{"id": 0, "qa_text": "{Zeus, Windows, vuln, indicator, org}"}]
        processed_qa = qa_analyzer.preprocess_qa_data(test_qa_data)
        
        if processed_qa:
            print("   ✅ QA Analyzer working")
        
        # Test Visualization Manager
        viz = VisualizationManager()
        if viz.output_dir.exists():
            print("   ✅ Visualization Manager working")
            
    except Exception as e:
        print(f"   ❌ Analyzers failed: {e}")


def test_full_pipeline():
    """전체 파이프라인 테스트 (API key 필요)"""
    if not Settings.validate_api_key():
        print("Skipping full pipeline test (no API key)")
        return
    
    print("=" * 50)
    print("Testing Full Pipeline")
    print("=" * 50)
    
    try:
        # Create test data
        malware_file, ner_file = create_test_data()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize pipeline
            pipeline = HallucinationVisionPipeline(temp_dir)
            
            # Run pipeline (with minimal data to save API calls)
            results = pipeline.run_complete_pipeline(
                malware_file_path=malware_file,
                original_ner_path=ner_file,
                sources=["threatpost"],
                tasks=["summarization"],  # Only summarization to save API calls
                wiki_paragraph_level=1    # Minimal data
            )
            
            if results and "scraped_data" in results:
                print("✅ Full pipeline test completed successfully")
                
                # Check output structure
                output_dir = Path(temp_dir)
                expected_dirs = ["scraped_data", "llm_results", "analysis", "figures"]
                
                for dir_name in expected_dirs:
                    if (output_dir / dir_name).exists():
                        print(f"   ✅ {dir_name}/ directory created")
                    else:
                        print(f"   ⚠️ {dir_name}/ directory missing")
                
                # Check for report
                if (output_dir / "PIPELINE_REPORT.md").exists():
                    print("   ✅ Pipeline report generated")
                
            else:
                print("⚠️ Full pipeline test completed with warnings")
        
        # Cleanup
        os.unlink(malware_file)
        os.unlink(ner_file)
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")


def test_legacy_compatibility():
    """레거시 호환성 테스트"""
    print("=" * 50)
    print("Testing Legacy Compatibility")
    print("=" * 50)
    
    try:
        # Test legacy data adapter
        from .data_scraper.adapter import DataAdapter
        
        adapter = DataAdapter()
        
        # Test scraped data conversion
        example_scraped = [{"malware_name": "zeus", "combined_content": "test content"}]
        text_list = adapter.scraped_to_text_list(example_scraped)
        
        if text_list:
            print("   ✅ Data adapter working")
        
        # Test legacy format conversion
        example_llm_results = [{
            "id": 0,
            "original_text": "test",
            "summary_text": "summary",
            "qa_text": "{a,b,c,d,e}"
        }]
        
        standardized = adapter.llm_results_to_standard_format(example_llm_results, "combined")
        
        if standardized:
            print("   ✅ Legacy format conversion working")
        
        print("✅ Legacy compatibility test completed")
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")


def main():
    """메인 테스트 함수"""
    print("🧪 HalluVision Pipeline Test Suite")
    print("=" * 60)
    
    # Check basic requirements
    print("Checking requirements...")
    
    # Check API key
    if Settings.validate_api_key():
        print("   ✅ OpenAI API key found")
    else:
        print("   ⚠️ OpenAI API key not found (some tests will be skipped)")
    
    # Check Python version
    import sys
    if sys.version_info >= (3, 8):
        print(f"   ✅ Python version: {sys.version}")
    else:
        print(f"   ❌ Python version {sys.version} < 3.8")
    
    print()
    
    # Run tests
    test_individual_components()
    print()
    
    test_legacy_compatibility()
    print()
    
    test_full_pipeline()
    print()
    
    print("=" * 60)
    print("🎉 Test suite completed!")
    print()
    print("Next steps:")
    print("1. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Prepare your malware list file")
    print("3. Run: python main.py your_malware_list.txt")


if __name__ == "__main__":
    main()
