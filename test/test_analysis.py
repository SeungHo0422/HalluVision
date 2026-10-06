#!/usr/bin/env python3
"""
기존 LLM 결과 데이터로 분석 및 평가 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path

import pytest
from analyzer.qa_analyzer import QAAnalyzer
from analyzer.visualization import VisualizationManager

def run_qa_analysis():
    """기존 데이터로 QA 분석 테스트"""
    
    print("="*60)
    print("QA Analysis Test with Existing Data")
    print("="*60)
    
    # 파일 경로 설정
    ground_truth_path = "datasets/ground_truth_ner.json"
    llm_results_path = "datasets/llm_results_gpt3.5_turbo_250427.json"
    output_dir = "test_analysis_output"
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)
    
    # QA Analyzer 초기화
    qa_analyzer = QAAnalyzer()
    
    print(f"\n1. 데이터 로딩:")
    print(f"   - Ground Truth: {ground_truth_path}")
    print(f"   - LLM Results: {llm_results_path}")
    
    # 기존 LLM 결과 데이터를 QA 분석 형태로 변환
    with open(llm_results_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    # QA 데이터 형태로 변환 (처음 100개만 테스트)
    qa_data_for_analysis = []
    max_items = 100  # 전체 데이터는 너무 크므로 일부만
    
    for i, item in enumerate(llm_data[:max_items]):
        qa_item = {
            "id": item.get("id"),
            "qa_text": item.get("qa_text", "")
        }
        qa_data_for_analysis.append(qa_item)
    
    print(f"\n2. 분석 준비:")
    print(f"   - 처리할 QA 항목: {len(qa_data_for_analysis)}개")
    print(f"   - 모델명: gpt-3.5-turbo")
    
    # Ground Truth 데이터 로드
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"   - Ground Truth 항목: {len(original_data)}개")
    
    print(f"\n3. QA 분석 실행:")
    try:
        # QA 평가 실행
        qa_results = qa_analyzer.run_full_evaluation(
            original_data,
            qa_data_for_analysis,
            model_name="gpt-3.5-turbo",
            save_failures=True,
            output_dir=output_dir
        )
        
        print(f"   ✅ QA 분석 완료!")
        
        # 결과 요약 출력
        if "metrics" in qa_results:
            metrics = qa_results["metrics"]
            
            if "overall_metrics" in metrics:
                om = metrics["overall_metrics"]
                print(f"\n4. 전체 성능 지표:")
                print(f"   - 평균 Precision: {om.get('avg_precision', 0):.4f}")
                print(f"   - 평균 Recall: {om.get('avg_recall', 0):.4f}")
                print(f"   - 평균 F1: {om.get('avg_f1', 0):.4f}")
            
            if "label_metrics" in metrics:
                print(f"\n5. 라벨별 성능:")
                for label, label_metrics in metrics["label_metrics"].items():
                    print(f"   - {label}: P={label_metrics.get('precision', 0):.4f}, "
                          f"R={label_metrics.get('recall', 0):.4f}, "
                          f"F1={label_metrics.get('f1', 0):.4f}")
        
        # 실패 분석
        if "failure_analysis" in qa_results:
            failure_analysis = qa_results["failure_analysis"]
            print(f"\n6. 실패 분석:")
            print(f"   - 총 실패 항목: {failure_analysis.get('total_failures', 0)}개")
            
            if "common_failure_patterns" in failure_analysis:
                print(f"   - 주요 실패 패턴:")
                for pattern, count in failure_analysis["common_failure_patterns"].items():
                    print(f"     * {pattern}: {count}회")
        
        # 결과 저장
        results_path = Path(output_dir) / "qa_analysis_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n7. 결과 저장:")
        print(f"   - 분석 결과: {results_path}")
        print(f"   - 출력 디렉토리: {output_dir}/")
        
        return qa_results
        
    except Exception as e:
        print(f"   ❌ QA 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

@pytest.fixture(scope="module", name="qa_results")
def _qa_results_fixture():
    return run_qa_analysis()


def test_qa_analysis(qa_results):
    assert qa_results, "QA analysis returned nothing"


def test_visualization(qa_results):
    """시각화 테스트"""
    
    if not qa_results:
        print("\n⚠️  QA 결과가 없어 시각화를 건너뜁니다.")
        return
    
    print(f"\n" + "="*60)
    print("Visualization Test")
    print("="*60)
    
    try:
        # 시각화 매니저 초기화
        visualizer = VisualizationManager("test_analysis_output/figures")
        
        # QA 성능 히트맵
        if "metrics" in qa_results:
            print("\n1. QA 성능 히트맵 생성...")
            qa_metrics = {"gpt-3.5-turbo": qa_results["metrics"]}
            visualizer.plot_qa_performance_heatmap(
                qa_metrics,
                "test_analysis_output/figures/qa_performance_heatmap.pdf"
            )
            print("   ✅ 히트맵 생성 완료")
        
        # 실패 분석 시각화
        if "failure_analysis" in qa_results:
            print("\n2. 실패 분석 차트 생성...")
            failure_analysis = qa_results["failure_analysis"]
            visualizer.plot_failure_analysis(
                failure_analysis,
                "test_analysis_output/figures/failure_analysis.pdf"
            )
            print("   ✅ 실패 분석 차트 생성 완료")
        
        print(f"\n3. 시각화 파일:")
        print(f"   - 디렉토리: test_analysis_output/figures/")
        print(f"   - QA 성능 히트맵: qa_performance_heatmap.pdf")
        print(f"   - 실패 분석: failure_analysis.pdf")
        
    except Exception as e:
        print(f"   ❌ 시각화 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 테스트 함수"""
    
    print("📊 기존 LLM 결과 데이터 분석 테스트 시작")
    print("🔍 gpt-3.5-turbo 결과와 ground truth 비교")
    
    # QA 분석 테스트
    qa_results = run_qa_analysis()
    
    # 시각화 테스트
    test_visualization(qa_results)
    
    print(f"\n" + "="*60)
    print("✅ 분석 테스트 완료!")
    print("📁 결과는 test_analysis_output/ 디렉토리에서 확인하세요.")
    print("="*60)

if __name__ == "__main__":
    main()
