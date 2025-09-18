#!/usr/bin/env python3
"""
기존 LLM 결과 데이터로 CDF 분석 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path
from analyzer.cdf_evaluator import CDFEvaluator

def test_cdf_analysis():
    """기존 데이터로 CDF 분석 테스트"""
    
    print("="*60)
    print("CDF Analysis Test with Existing Data")
    print("="*60)
    
    # 파일 경로 설정
    ground_truth_path = "datasets/ground_truth_ner.json"
    llm_results_35_path = "datasets/llm_results_gpt3.5_turbo_250427.json"
    llm_results_4o_path = "datasets/llm_results_gpt4o_mini_2503.json"
    
    print(f"\n1. 데이터 로딩:")
    print(f"   - Ground Truth: {ground_truth_path}")
    print(f"   - GPT-3.5-turbo: {llm_results_35_path}")
    print(f"   - GPT-4o-mini: {llm_results_4o_path}")
    
    # 데이터 로드
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(llm_results_35_path, 'r', encoding='utf-8') as f:
        llm_results_35 = json.load(f)
    
    with open(llm_results_4o_path, 'r', encoding='utf-8') as f:
        llm_results_4o = json.load(f)
    
    # 샘플 크기 제한 (전체 데이터는 너무 큼)
    sample_size = 500
    llm_results_35_sample = llm_results_35[:sample_size]
    llm_results_4o_sample = llm_results_4o[:sample_size]
    
    print(f"\n2. 분석 데이터:")
    print(f"   - Ground Truth 항목: {len(original_data)}개")
    print(f"   - GPT-3.5-turbo 샘플: {len(llm_results_35_sample)}개")
    print(f"   - GPT-4o-mini 샘플: {len(llm_results_4o_sample)}개")
    
    # CDF 평가자 초기화
    evaluator = CDFEvaluator("cdf_analysis_output")
    
    # 개별 모델 분석
    print(f"\n3. 개별 모델 CDF 분석:")
    
    # GPT-3.5-turbo 분석
    print("\n🔍 GPT-3.5-turbo 분석...")
    report_35 = evaluator.generate_metrics_cdf_report(
        original_data=original_data,
        qa_results=llm_results_35_sample,
        model_name="gpt-3.5-turbo"
    )
    
    # GPT-4o-mini 분석
    print("\n🔍 GPT-4o-mini 분석...")
    report_4o = evaluator.generate_metrics_cdf_report(
        original_data=original_data,
        qa_results=llm_results_4o_sample,
        model_name="gpt-4o-mini"
    )
    
    # 모델 비교 분석
    print(f"\n4. 모델 비교 Precision/Recall/F1 CDF 분석:")
    
    # 모델 결과 딕셔너리
    model_results = {
        "gpt-3.5-turbo": llm_results_35_sample,
        "gpt-4o-mini": llm_results_4o_sample
    }
    
    # 비교 CDF 그래프 생성
    comparison_stats = evaluator.plot_comparative_metrics_cdf(
        original_data, model_results, "qa"
    )
    
    # 결과 요약
    print(f"\n5. 결과 요약:")
    print("="*40)
    
    if report_35.get("qa_analysis"):
        qa_35 = report_35["qa_analysis"]
        print(f"📊 GPT-3.5-turbo QA:")
        for metric in ["precision", "recall", "f1"]:
            if metric in qa_35["metrics"]:
                metric_stats = qa_35["metrics"][metric]
                print(f"   📈 {metric.upper()}: 전체 평균 {metric_stats['overall_mean']:.4f}")
                best_label = max(metric_stats['mean_scores'], key=metric_stats['mean_scores'].get)
                worst_label = min(metric_stats['mean_scores'], key=metric_stats['mean_scores'].get)
                print(f"      - 최고: {best_label} ({metric_stats['mean_scores'][best_label]:.4f})")
                print(f"      - 최저: {worst_label} ({metric_stats['mean_scores'][worst_label]:.4f})")
    
    if report_4o.get("qa_analysis"):
        qa_4o = report_4o["qa_analysis"]
        print(f"\n📊 GPT-4o-mini QA:")
        for metric in ["precision", "recall", "f1"]:
            if metric in qa_4o["metrics"]:
                metric_stats = qa_4o["metrics"][metric]
                print(f"   📈 {metric.upper()}: 전체 평균 {metric_stats['overall_mean']:.4f}")
                best_label = max(metric_stats['mean_scores'], key=metric_stats['mean_scores'].get)
                worst_label = min(metric_stats['mean_scores'], key=metric_stats['mean_scores'].get)
                print(f"      - 최고: {best_label} ({metric_stats['mean_scores'][best_label]:.4f})")
                print(f"      - 최저: {worst_label} ({metric_stats['mean_scores'][worst_label]:.4f})")
    
    # 모델 비교
    if len(comparison_stats) >= 2:
        print(f"\n📈 모델 성능 비교 (F1-Score 기준):")
        models = list(comparison_stats.keys())
        for label in evaluator.LABELS:
            if ("f1" in comparison_stats[models[0]] and 
                label in comparison_stats[models[0]]["f1"] and
                "f1" in comparison_stats[models[1]] and 
                label in comparison_stats[models[1]]["f1"]):
                
                score_35 = comparison_stats[models[0]]["f1"][label]["mean"]
                score_4o = comparison_stats[models[1]]["f1"][label]["mean"]
                better_model = models[0] if score_35 > score_4o else models[1]
                print(f"   - {label}: {better_model} 우수 ({max(score_35, score_4o):.4f} vs {min(score_35, score_4o):.4f})")
    
    print(f"\n✅ CDF 분석 완료!")
    print(f"📁 결과는 cdf_analysis_output/ 디렉토리에서 확인하세요.")
    
    return report_35, report_4o, comparison_stats

def main():
    """메인 테스트 함수"""
    
    print("📊 기존 LLM 결과 데이터 CDF 분석 테스트 시작")
    print("🔍 GPT-3.5-turbo와 GPT-4o-mini 성능 비교")
    
    try:
        report_35, report_4o, comparison_stats = test_cdf_analysis()
        
        print(f"\n" + "="*60)
        print("✅ Precision/Recall/F1-score CDF 분석 테스트 완료!")
        print("📊 생성된 CDF 그래프:")
        print("   - qa_metrics_cdf_gpt_3_5_turbo.pdf")
        print("   - qa_metrics_cdf_gpt_4o_mini.pdf") 
        print("   - qa_comparative_metrics_cdf.pdf")
        print("📝 분석 리포트:")
        print("   - metrics_cdf_report_gpt_3_5_turbo.json")
        print("   - metrics_cdf_report_gpt_4o_mini.json")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
