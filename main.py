"""
Main Pipeline Module

통합된 HalluVision 파이프라인
Data Scraper -> LLM Proxy -> Preprocessor -> Analyzer 전체 워크플로우 제공
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from datetime import datetime

# Import pipeline components
from data_scraper.main import DataScraper
from data_scraper.adapter import DataAdapter
from llm_proxy.summarizer import AbstractiveSummarizer
from llm_proxy.qa_handler import QuestionAnsweringHandler
from preprocessor.summary_preprocessor import SummaryPreprocessor
from preprocessor.qa_preprocessor import QAPreprocessor
from analyzer.summary_analyzer import SummaryAnalyzer
from analyzer.qa_analyzer import QAAnalyzer
from analyzer.visualization import VisualizationManager
from utils.id_mapper import create_compatible_results
from config import Settings

class HallucinationVisionPipeline:
    """
    전체 HalluVision 파이프라인 관리 클래스
    모든 모듈을 통합하여 end-to-end 처리 제공
    """
    
    def __init__(
        self, 
        output_dir: str = "pipeline_output",
        api_key: Optional[str] = None
    ):
        """
        Initialize the complete pipeline
        
        Args:
            output_dir (str): Output directory for all results
            api_key (Optional[str]): OpenAI API key
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "scraped_data").mkdir(exist_ok=True)
        (self.output_dir / "llm_results").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Initialize components
        self.data_scraper = DataScraper(str(self.output_dir / "scraped_data"))
        self.data_adapter = DataAdapter()
        
        # LLM components
        model = Settings.get_model_for_task("summarization")
        self.summarizer = AbstractiveSummarizer(model, api_key)
        self.qa_handler = QuestionAnsweringHandler(model, api_key)
        
        # Preprocessors
        self.summary_preprocessor = SummaryPreprocessor()
        self.qa_preprocessor = QAPreprocessor()
        
        # Analyzers
        self.summary_analyzer = SummaryAnalyzer()
        self.qa_analyzer = QAAnalyzer()
        self.visualizer = VisualizationManager(str(self.output_dir / "figures"))
        
        # Pipeline state
        self.scraped_data = None
        self.llm_results = None
        self.analysis_results = {}
        
        print(f"Pipeline initialized. Output directory: {self.output_dir}")
    
    def run_data_scraping(
        self, 
        malware_file_path: str, 
        sources: List[str] = ["threatpost"],
        wiki_paragraph_level: int = 3,
        threatpost_urls_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        1단계: 데이터 스크래핑 실행
        
        Args:
            malware_file_path (str): Path to malware list file
            sources (List[str]): Data sources to use
            wiki_paragraph_level (int): Wikipedia paragraph level
            threatpost_urls_file (Optional[str]): Threatpost URL file path
            
        Returns:
            List[Dict[str, Any]]: Scraped data
        """
        print("=" * 50)
        print("STEP 1: Data Scraping")
        print("=" * 50)
        
        # Run scraping
        self.scraped_data = self.data_scraper.scrape_from_file(
            malware_file_path, sources, wiki_paragraph_level, threatpost_urls_file
        )
        # print("-------[DEBUG: scraped_data]-------\n", self.scraped_data)
        # Save scraped data
        scraped_path = self.output_dir / "scraped_data" / "scraped_data.json"
        self.data_scraper.save_results(self.scraped_data, str(scraped_path))
        
        # Print statistics
        stats = self.data_scraper.get_statistics(self.scraped_data)
        print(f"\nScraping completed:")
        print(f"  - Total malware processed: {stats['total_malware']}")
        print(f"  - Total paragraphs collected: {stats['total_paragraphs']}")
        
        return self.scraped_data
    
    def run_llm_processing(
        self, 
        tasks: List[str] = ["summarization", "qa"],
        prompt_version: str = "v2"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        2단계: LLM 처리 실행 (요약 및 QA)
        
        Args:
            tasks (List[str]): Tasks to perform ("summarization", "qa")
            prompt_version (str): Prompt version for summarization
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: LLM processing results
        """
        if self.scraped_data is None:
            raise ValueError("No scraped data available. Run data scraping first.")
        
        print("=" * 50)
        print("STEP 2: LLM Processing")
        print("=" * 50)
        
        results = {}
        
        # Summarization
        if "summarization" in tasks:
            print("Running abstractive summarization...")
            summary_results = self.summarizer.summarize_scraped_data(
                self.scraped_data, prompt_version
            )
            results["summarization"] = summary_results
            
            # Save summary results
            summary_path = self.output_dir / "llm_results" / "summary_results.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_results, f, ensure_ascii=False, indent=2)
            
            print(f"  - Summarization completed: {len(summary_results)} items")
        
        # Question Answering
        if "qa" in tasks:
            print("Running question answering...")
            qa_results = self.qa_handler.qa_scraped_data(self.scraped_data)
            results["qa"] = qa_results
            
            # Save QA results
            qa_path = self.output_dir / "llm_results" / "qa_results.json"
            with open(qa_path, 'w', encoding='utf-8') as f:
                json.dump(qa_results, f, ensure_ascii=False, indent=2)
            
            print(f"  - QA processing completed: {len(qa_results)} items")
        
        # Save in legacy format for compatibility
        if "summarization" in results and "qa" in results:
            # Create combined results
            combined_results = []
            
            for i, (sum_res, qa_res) in enumerate(zip(results["summarization"], results["qa"])):
                combined = {
                    "id": i,
                    "malware_name": sum_res.get("malware_name"),
                    "original_text": sum_res.get("original_content"),
                    "summary_text": sum_res.get("summary"),
                    "qa_text": qa_res.get("qa_text"),
                    "parsed_qa": qa_res.get("parsed_qa"),
                    "model": sum_res.get("model")
                }
                combined_results.append(combined)
            
            # Save in legacy format
            self.data_adapter.save_in_legacy_format(
                combined_results, 
                str(self.output_dir / "llm_results" / "gpt_results")
            )
        
        self.llm_results = results
        return results
    
    def run_analysis(
        self, 
        original_ner_path: Optional[str] = None,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        3단계: 분석 실행 (전처리 + 평가)
        
        Args:
            original_ner_path (Optional[str]): Path to original NER data for evaluation
            model_names (Optional[List[str]]): Names for different models being compared
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if self.llm_results is None:
            raise ValueError("No LLM results available. Run LLM processing first.")
        
        print("=" * 50)
        print("STEP 3: Analysis & Evaluation")
        print("=" * 50)
        
        analysis_results = {}
        
        # Load original NER data if provided
        original_data = {}
        if original_ner_path and os.path.exists(original_ner_path):
            try:
                with open(original_ner_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                print(f"Loaded original NER data: {len(original_data)} documents")
            except Exception as e:
                print(f"Warning: Could not load original NER data: {e}")
        
        # Summary Analysis
        if "summarization" in self.llm_results:
            print("Analyzing summarization results...")
            
            # This would require NER processing on summaries
            # For now, we'll create a placeholder
            summary_metrics = {
                "total_summaries": len(self.llm_results["summarization"]),
                "avg_summary_length": sum(len(r.get("summary", "").split()) 
                                        for r in self.llm_results["summarization"]) / len(self.llm_results["summarization"])
            }
            
            analysis_results["summary"] = summary_metrics
            print(f"  - Summary analysis completed")
        
        # QA Analysis with ID Mapping
        if "qa" in self.llm_results and original_data:
            print("Analyzing QA results...")
            
            # Create ground truth compatible results using ID mapping
            try:
                print("  - Creating ground truth compatible results with ID mapping...")
                
                # 임시 파일 경로
                temp_qa_path = str(self.output_dir / "llm_results" / "qa_results.json")
                mapped_qa_path = str(self.output_dir / "llm_results" / "qa_results_mapped.json")
                gt_path = "datasets/ground_truth_ner.json"
                existing_llm_path = "datasets/llm_results_gpt4o_mini_2503.json"
                
                # ID 매핑 수행
                mapping_report_path = create_compatible_results(
                    pipeline_results_path=temp_qa_path,
                    ground_truth_path=gt_path,
                    existing_llm_results_path=existing_llm_path,
                    output_path=mapped_qa_path
                )
                
                # 매핑된 결과로 QA 데이터 준비
                with open(mapped_qa_path, 'r', encoding='utf-8') as f:
                    mapped_qa_results = json.load(f)
                
                qa_data_for_analysis = []
                for result in mapped_qa_results:
                    qa_item = {
                        "id": result.get("id"),
                        "qa_text": result.get("qa_text", "")
                    }
                    qa_data_for_analysis.append(qa_item)
                
                print(f"  - Successfully mapped {len(qa_data_for_analysis)} QA items to ground truth IDs")
                
            except Exception as e:
                print(f"  - Warning: ID mapping failed ({e}), using sequential IDs...")
                # 백업: 순차적 ID 사용
                qa_data_for_analysis = []
                for i, result in enumerate(self.llm_results["qa"]):
                    qa_item = {
                        "id": i,
                        "qa_text": result.get("qa_text", "")
                    }
                    qa_data_for_analysis.append(qa_item)
            
            # Run QA evaluation
            qa_results = self.qa_analyzer.run_full_evaluation(
                original_data,
                qa_data_for_analysis,
                model_names[0] if model_names else "pipeline_model",
                save_failures=True,
                output_dir=str(self.output_dir / "analysis")
            )
            
            analysis_results["qa"] = qa_results
            print(f"  - QA analysis completed")
        
        # Save analysis results
        analysis_path = self.output_dir / "analysis" / "analysis_results.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def generate_visualizations(self):
        """
        4단계: 시각화 생성
        """
        if not self.analysis_results:
            print("No analysis results available for visualization.")
            return
        
        print("=" * 50)
        print("STEP 4: Generating Visualizations")
        print("=" * 50)
        
        # Generate QA performance visualizations if available
        if "qa" in self.analysis_results:
            qa_metrics = self.analysis_results["qa"]["metrics"]
            failure_analysis = self.analysis_results["qa"]["failure_analysis"]
            
            # QA performance heatmap
            self.visualizer.plot_qa_performance_heatmap(
                {"model": qa_metrics},
                str(self.output_dir / "figures" / "qa_performance_heatmap.pdf")
            )
            
            # Failure analysis
            self.visualizer.plot_failure_analysis(
                failure_analysis,
                str(self.output_dir / "figures" / "failure_analysis.pdf")
            )
            
            print("  - QA visualizations generated")
        
        print("  - Visualization generation completed")
    
    def run_complete_pipeline(
        self,
        malware_file_path: str,
        original_ner_path: Optional[str] = None,
        sources: List[str] = ["threatpost"],
        tasks: List[str] = ["summarization", "qa"],
        wiki_paragraph_level: int = 3,
        prompt_version: str = "v2",
        threatpost_urls_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Args:
            malware_file_path (str): Path to malware list file
            original_ner_path (Optional[str]): Path to original NER data
            sources (List[str]): Data sources
            tasks (List[str]): LLM tasks to perform
            wiki_paragraph_level (int): Wikipedia paragraph level
            prompt_version (str): Prompt version
            threatpost_urls_file (Optional[str]): Threatpost URL file path
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        start_time = datetime.now()
        
        print("Starting HalluVision Pipeline")
        print(f"Start time: {start_time}")
        print()
        
        try:
            # Step 1: Data Scraping
            scraped_data = self.run_data_scraping(
                malware_file_path, sources, wiki_paragraph_level, threatpost_urls_file
            )
            
            # Step 2: LLM Processing
            llm_results = self.run_llm_processing(tasks, prompt_version)
            
            # Step 3: Analysis
            analysis_results = self.run_analysis(original_ner_path)
            
            # Step 4: Visualizations
            self.generate_visualizations()
            
            # Create final report
            self.create_final_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print()
            print("Pipeline completed successfully!")
            print(f"Total duration: {duration}")
            print(f"Results saved in: {self.output_dir}")
            
            return {
                "scraped_data": scraped_data,
                "llm_results": llm_results,
                "analysis_results": analysis_results,
                "duration": str(duration),
                "output_dir": str(self.output_dir)
            }
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise
    
    def create_final_report(self):
        """최종 리포트 생성"""
        report_path = self.output_dir / "PIPELINE_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# HalluVision Pipeline Report\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # Data Scraping Section
            if self.scraped_data:
                stats = self.data_scraper.get_statistics(self.scraped_data)
                f.write("## Data Scraping Results\n\n")
                f.write(f"- **Total malware processed**: {stats['total_malware']}\n")
                f.write(f"- **Total paragraphs collected**: {stats['total_paragraphs']}\n")
                f.write(f"- **Source statistics**: {stats['source_stats']}\n\n")
            
            # LLM Processing Section
            if self.llm_results:
                f.write("## LLM Processing Results\n\n")
                for task, results in self.llm_results.items():
                    f.write(f"- **{task.capitalize()}**: {len(results)} items processed\n")
                f.write("\n")
            
            # Analysis Section
            if self.analysis_results:
                f.write("## Analysis Results\n\n")
                
                if "qa" in self.analysis_results:
                    qa_metrics = self.analysis_results["qa"]["metrics"]
                    f.write("### QA Performance\n\n")
                    
                    if "overall_metrics" in qa_metrics:
                        om = qa_metrics["overall_metrics"]
                        f.write(f"- **Overall Precision**: {om['avg_precision']:.4f}\n")
                        f.write(f"- **Overall Recall**: {om['avg_recall']:.4f}\n")
                        f.write(f"- **Overall F1**: {om['avg_f1']:.4f}\n\n")
                    
                    if "label_metrics" in qa_metrics:
                        f.write("#### Label-wise Performance\n\n")
                        for label, metrics in qa_metrics["label_metrics"].items():
                            f.write(f"- **{label}**: P={metrics['precision']:.4f}, "
                                   f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}\n")
                        f.write("\n")
            
            # File Structure
            f.write("## Output Structure\n\n")
            f.write("```\n")
            f.write(f"{self.output_dir}/\n")
            f.write("├── scraped_data/           # Raw scraped data\n")
            f.write("├── llm_results/           # LLM processing results\n")
            f.write("├── analysis/              # Analysis and evaluation results\n")
            f.write("├── figures/               # Generated visualizations\n")
            f.write("└── PIPELINE_REPORT.md     # This report\n")
            f.write("```\n\n")
        
        print(f"📋 Final report generated: {report_path}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="HalluVision Pipeline")
    
    parser.add_argument("--malware-file", help="Path to malware list file")
    parser.add_argument("--original-ner", help="Path to original NER data for evaluation")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory")
    parser.add_argument("--sources", nargs="+", default=["threatpost"], 
                       help="Data sources to use")
    parser.add_argument("--tasks", nargs="+", default=["summarization", "qa"],
                       help="LLM tasks to perform")
    parser.add_argument("--wiki-paragraphs", type=int, default=3,
                       help="Number of Wikipedia paragraphs to extract")
    parser.add_argument("--prompt-version", default="v2",
                       help="Prompt version for summarization")
    parser.add_argument("--threatpost-urls", 
                       help="Path to Threatpost URLs file (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Check API key
    if not Settings.validate_api_key():
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    # Initialize and run pipeline
    pipeline = HallucinationVisionPipeline(args.output_dir)
    
    pipeline.run_complete_pipeline(
        malware_file_path=args.malware_file,
        original_ner_path=args.original_ner,
        sources=args.sources,
        tasks=args.tasks,
        wiki_paragraph_level=args.wiki_paragraphs,
        prompt_version=args.prompt_version,
        threatpost_urls_file=args.threatpost_urls
    )


if __name__ == "__main__":
    main()
