"""
CDF 기반 평가 클래스
references의 analysis_qa.ipynb와 eval_summary_prof.ipynb를 기반으로 구현
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


class CDFEvaluator:
    """CDF 기반 성능 평가 클래스"""
    
    def __init__(self, output_dir: str = "cdf_analysis_output"):
        """
        Args:
            output_dir (str): 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 그래프 저장 디렉토리
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # NER 라벨
        self.LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
        self.LABELS_DISPLAY = ['MAL', 'SYS', 'IND', 'VUL', 'ORG']
        
        # 모델별 색상
        self.model_colors = {
            "gpt-3.5-turbo": "orange",
            "gpt-4o-mini": "green",
            "pipeline_model": "blue"
        }
    
    def calculate_entity_metrics_scores(
        self, 
        original_data: Dict[str, Any], 
        llm_results: List[Dict[str, Any]],
        metric_type: str = "f1"
    ) -> np.ndarray:
        """
        엔티티 매트릭 점수 계산 (precision, recall, f1-score 기반)
        
        Args:
            original_data: Ground truth 데이터
            llm_results: LLM 결과 데이터
            metric_type: 'precision', 'recall', 'f1' 중 하나
            
        Returns:
            np.ndarray: metrics matrix [문서수, 라벨수]
        """
        print(f"  - 엔티티 {metric_type.upper()} 점수 계산 중...")
        
        # 결과를 ID별로 매핑
        llm_by_id = {item["id"]: item for item in llm_results}
        
        metrics_matrix = []
        
        for doc_id in range(len(llm_results)):
            doc_id_str = str(doc_id)
            
            # Ground truth 데이터 확인
            if doc_id_str not in original_data:
                continue
                
            original = original_data[doc_id_str]
            
            # LLM 결과 확인
            if doc_id not in llm_by_id:
                continue
                
            pred_item = llm_by_id[doc_id]
            
            doc_scores = []
            
            for label in self.LABELS:
                # Ground truth 엔티티
                true_entities = set(entity.lower() for entity in original.get(label, []))
                
                # 예측 엔티티 (QA 또는 요약 결과에서 추출)
                pred_entities = set()
                if "qa_text" in pred_item:
                    # QA 결과에서 엔티티 추출
                    qa_text = pred_item["qa_text"].strip("{} ")
                    if qa_text and qa_text.lower() != "no answer":
                        entities = [e.strip() for e in qa_text.split(",")]
                        if len(entities) == len(self.LABELS):
                            # 순서대로 매핑 (references 코드 참조)
                            label_idx = self.LABELS.index(label)
                            if label_idx == 2:  # Indicator
                                entity = entities[3].lower()  # Vulnerability와 교체
                            elif label_idx == 3:  # Vulnerability
                                entity = entities[2].lower()  # Indicator와 교체
                            else:
                                entity = entities[label_idx].lower()
                            
                            if entity != "no answer":
                                pred_entities.add(entity)
                
                elif "summary_text" in pred_item:
                    # 요약 결과에서는 별도 처리 필요
                    # 현재는 간단하게 처리
                    summary_text = pred_item.get("summary_text", "").lower()
                    for entity in true_entities:
                        if entity in summary_text:
                            pred_entities.add(entity)
                
                # Precision, Recall, F1-score 계산
                if len(true_entities) == 0 and len(pred_entities) == 0:
                    precision = recall = f1 = 1.0
                elif len(true_entities) == 0:
                    precision = recall = f1 = 0.0
                elif len(pred_entities) == 0:
                    precision = recall = f1 = 0.0
                else:
                    tp = len(true_entities & pred_entities)  # True Positives
                    fp = len(pred_entities - true_entities)  # False Positives  
                    fn = len(true_entities - pred_entities)  # False Negatives
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # 선택된 메트릭 반환
                if metric_type == "precision":
                    score = precision
                elif metric_type == "recall":
                    score = recall
                else:  # f1
                    score = f1
                
                doc_scores.append(score)
            
            metrics_matrix.append(doc_scores)
        
        return np.array(metrics_matrix)
    
    def plot_metrics_cdf(
        self, 
        original_data: Dict[str, Any],
        llm_results: List[Dict[str, Any]], 
        model_name: str, 
        plot_type: str = "qa"
    ) -> Dict[str, Any]:
        """
        Precision, Recall, F1-score CDF 그래프 생성
        
        Args:
            original_data: Ground truth 데이터
            llm_results: LLM 결과 데이터
            model_name: 모델명
            plot_type: 'qa' 또는 'summary'
            
        Returns:
            Dict: 통계 정보
        """
        print(f"  - {plot_type.upper()} Precision/Recall/F1 CDF 그래프 생성 중...")
        
        metrics = ["precision", "recall", "f1"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        all_stats = {}
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 메트릭 매트릭스 계산
            metrics_matrix = self.calculate_entity_metrics_scores(
                original_data, llm_results, metric
            )
            
            num_labels = metrics_matrix.shape[1]
            label_median = []
            label_mean = []
            
            for i in range(num_labels):
                label_scores = metrics_matrix[:, i]
                sorted_scores = np.sort(label_scores)
                
                label_median.append(np.median(sorted_scores))
                label_mean.append(np.mean(sorted_scores))
                
                # CDF 계산
                cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                
                # 플롯
                ax.plot(sorted_scores, cdf, 
                       label=f"{self.LABELS_DISPLAY[i]} (μ: {np.mean(sorted_scores):.3f})",
                       linewidth=2, alpha=0.8)
            
            ax.set_xlabel(f"{metric.capitalize()} Score", fontsize=12)
            ax.set_ylabel("Cumulative Probability", fontsize=12)
            ax.set_title(f"{metric.capitalize()} CDF", fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # "Better" 화살표 추가
            ax.annotate(
                text='Better →',
                xy=(0.7, 0.3),
                xytext=(0.5, 0.2),
                xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red'
            )
            
            # 통계 저장
            all_stats[metric] = {
                "median_scores": {label: median for label, median in zip(self.LABELS, label_median)},
                "mean_scores": {label: mean for label, mean in zip(self.LABELS, label_mean)},
                "overall_median": np.median(label_median),
                "overall_mean": np.mean(label_mean)
            }
        
        plt.suptitle(f"{plot_type.upper()} Performance CDF - {model_name}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 저장
        filename = f"{plot_type}_metrics_cdf_{model_name.replace('-', '_').replace('.', '_')}.pdf"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 통계 정보 반환
        stats = {
            "model_name": model_name,
            "plot_type": plot_type,
            "metrics": all_stats
        }
        
        print(f"    ✅ 저장됨: {filepath}")
        return stats
    
    def plot_comparative_metrics_cdf(
        self, 
        original_data: Dict[str, Any],
        model_results: Dict[str, List[Dict[str, Any]]], 
        plot_type: str = "qa"
    ) -> Dict[str, Any]:
        """
        여러 모델 비교 CDF 그래프 (Precision, Recall, F1-score)
        
        Args:
            original_data: Ground truth 데이터
            model_results: {model_name: llm_results} 딕셔너리
            plot_type: 'qa' 또는 'summary'
            
        Returns:
            Dict: 비교 통계 정보
        """
        print(f"  - {plot_type.upper()} 모델 비교 Precision/Recall/F1 CDF 그래프 생성 중...")
        
        metrics = ["precision", "recall", "f1"]
        fig, axes = plt.subplots(len(metrics), len(self.LABELS), figsize=(20, 12))
        
        if len(metrics) == 1:
            axes = axes.reshape(1, -1)
        if len(self.LABELS) == 1:
            axes = axes.reshape(-1, 1)
        
        comparison_stats = {}
        
        for metric_idx, metric in enumerate(metrics):
            for label_idx, label in enumerate(self.LABELS):
                ax = axes[metric_idx, label_idx]
                
                for model_name, llm_results in model_results.items():
                    # 메트릭 계산
                    metrics_matrix = self.calculate_entity_metrics_scores(
                        original_data, llm_results, metric
                    )
                    
                    if metrics_matrix.shape[1] > label_idx:
                        label_scores = metrics_matrix[:, label_idx]
                        sorted_scores = np.sort(label_scores)
                        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                        
                        color = self.model_colors.get(model_name, 'black')
                        ax.plot(sorted_scores, cdf, 
                               label=f"{model_name} (μ: {np.mean(sorted_scores):.3f})",
                               color=color, linewidth=2, alpha=0.8)
                        
                        # 통계 저장
                        if model_name not in comparison_stats:
                            comparison_stats[model_name] = {}
                        if metric not in comparison_stats[model_name]:
                            comparison_stats[model_name][metric] = {}
                        comparison_stats[model_name][metric][label] = {
                            "mean": np.mean(sorted_scores),
                            "median": np.median(sorted_scores),
                            "std": np.std(sorted_scores)
                        }
                
                ax.set_xlabel(f"{metric.capitalize()} Score", fontsize=10)
                if label_idx == 0:
                    ax.set_ylabel("Cumulative Probability", fontsize=10)
                
                title = f"{self.LABELS_DISPLAY[label_idx]}"
                if metric_idx == 0:
                    title = f"{title}\n{metric.capitalize()}"
                else:
                    title = metric.capitalize()
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"{plot_type.upper()} Performance Comparison - All Models & Metrics", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 저장
        filename = f"{plot_type}_comparative_metrics_cdf.pdf"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ 저장됨: {filepath}")
        return comparison_stats
    
    def generate_metrics_cdf_report(
        self, 
        original_data: Dict[str, Any],
        qa_results: Optional[List[Dict[str, Any]]] = None,
        summary_results: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "pipeline_model"
    ) -> Dict[str, Any]:
        """
        전체 Precision/Recall/F1-score CDF 분석 리포트 생성
        
        Args:
            original_data: Ground truth 데이터
            qa_results: QA 결과 데이터
            summary_results: 요약 결과 데이터
            model_name: 모델명
            
        Returns:
            Dict: 전체 분석 결과
        """
        print(f"\n📊 {model_name} Precision/Recall/F1-score CDF 분석 리포트 생성")
        print("="*60)
        
        report = {
            "model_name": model_name,
            "analysis_date": str(Path.cwd()),
            "qa_analysis": None,
            "summary_analysis": None
        }
        
        # QA 분석
        if qa_results:
            print("\n1. QA 성능 Precision/Recall/F1 CDF 분석:")
            qa_stats = self.plot_metrics_cdf(original_data, qa_results, model_name, "qa")
            report["qa_analysis"] = qa_stats
            
            print("   [QA 성능 요약]")
            for metric, metric_stats in qa_stats["metrics"].items():
                print(f"   📊 {metric.upper()}:")
                for label, score in metric_stats["mean_scores"].items():
                    print(f"      - {label}: {score:.4f}")
                print(f"      - 전체 평균: {metric_stats['overall_mean']:.4f}")
                print()
        
        # Summary 분석
        if summary_results:
            print("\n2. Summary 성능 Precision/Recall/F1 CDF 분석:")
            summary_stats = self.plot_metrics_cdf(original_data, summary_results, model_name, "summary")
            report["summary_analysis"] = summary_stats
            
            print("   [Summary 성능 요약]")
            for metric, metric_stats in summary_stats["metrics"].items():
                print(f"   📊 {metric.upper()}:")
                for label, score in metric_stats["mean_scores"].items():
                    print(f"      - {label}: {score:.4f}")
                print(f"      - 전체 평균: {metric_stats['overall_mean']:.4f}")
                print()
        
        # 리포트 저장
        report_path = self.output_dir / f"metrics_cdf_report_{model_name.replace('-', '_')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nCDF 분석 완료!")
        print(f"결과 저장: {self.output_dir}")
        print(f"그래프: {self.figures_dir}")
        print(f"리포트: {report_path}")
        
        return report


def main():
    """테스트용 메인 함수"""
    evaluator = CDFEvaluator()
    
    # 테스트 데이터 로드
    print("CDF Evaluator 테스트")
    print("테스트 데이터를 로드하고 CDF 그래프를 생성합니다...")


if __name__ == "__main__":
    main()
