"""
ID Mapping Module for Ground Truth and Pipeline Results
기존 ground truth와 새 파이프라인 결과 간의 ID 매핑 관리
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
from difflib import SequenceMatcher


class IDMapper:
    """
    기존 ground truth와 새 파이프라인 결과 간의 ID 매핑을 관리하는 클래스
    """
    
    def __init__(self):
        self.url_to_gt_mapping = {}  # URL -> Ground Truth ID 매핑
        self.gt_to_url_mapping = {}  # Ground Truth ID -> URL 매핑
        self.pipeline_to_gt_mapping = {}  # Pipeline ID -> Ground Truth ID 매핑
        
    def extract_url_from_content(self, content: str) -> Optional[str]:
        """
        텍스트 내용에서 Threatpost URL을 추출
        
        Args:
            content (str): 분석할 텍스트 내용
            
        Returns:
            Optional[str]: 추출된 URL (없으면 None)
        """
        # Threatpost URL 패턴 매칭
        threatpost_patterns = [
            r'https?://threatpost\.com/[^/]+/\d+/?',
            r'threatpost\.com/[^/]+/\d+/?'
        ]
        
        for pattern in threatpost_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                url = matches[0]
                # HTTP 프로토콜 정규화
                if not url.startswith('http'):
                    url = 'https://' + url
                # 마지막 슬래시 제거
                url = url.rstrip('/')
                return url
        
        return None
    
    def build_gt_url_mapping(self, ground_truth_path: str, llm_results_path: str):
        """
        기존 ground truth와 LLM 결과에서 URL 매핑 구축
        
        Args:
            ground_truth_path (str): Ground truth JSON 파일 경로
            llm_results_path (str): 기존 LLM 결과 JSON 파일 경로
        """
        print("[INFO] Ground Truth URL 매핑을 구축하는 중...")
        
        # LLM 결과에서 URL 추출
        with open(llm_results_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        
        url_count = 0
        for item in llm_data:
            item_id = item.get('id')
            original_text = item.get('original_text', '')
            
            # URL 추출
            url = self.extract_url_from_content(original_text)
            if url:
                self.url_to_gt_mapping[url] = item_id
                self.gt_to_url_mapping[item_id] = url
                url_count += 1
        
        print(f"[INFO] {url_count}개의 URL-ID 매핑을 구축했습니다.")
        
    def create_content_similarity_mapping(self, ground_truth_path: str, llm_results_path: str):
        """
        콘텐츠 유사도를 기반으로 한 매핑 생성
        
        Args:
            ground_truth_path (str): Ground truth JSON 파일 경로  
            llm_results_path (str): 기존 LLM 결과 JSON 파일 경로
        """
        print("[INFO] 콘텐츠 유사도 매핑을 구축하는 중...")
        
        with open(llm_results_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
            
        self.content_to_id_mapping = {}
        self.id_to_content_mapping = {}
        
        for item in llm_data:
            item_id = item.get('id')
            original_text = item.get('original_text', '')
            
            # 텍스트 정규화 (공백, 대소문자 정리)
            normalized_text = self._normalize_text(original_text)
            
            # 텍스트의 시작 부분과 끝 부분을 키로 사용 (빠른 검색용)
            text_signature = self._create_text_signature(normalized_text)
            
            self.content_to_id_mapping[text_signature] = item_id
            self.id_to_content_mapping[item_id] = normalized_text
                
        print(f"[INFO] {len(self.content_to_id_mapping)}개의 콘텐츠 유사도 매핑을 생성했습니다.")
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 공백 정리, 소문자 변환
        text = re.sub(r'\s+', ' ', text.strip().lower())
        # 특수문자 정리
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _create_text_signature(self, text: str, length: int = 100) -> str:
        """텍스트 시그니처 생성 (시작 부분)"""
        return text[:length] if len(text) > length else text
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 유사도 계산"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def find_best_content_match(self, new_text: str, threshold: float = 0.7) -> Optional[int]:
        """
        새로운 텍스트와 가장 유사한 기존 콘텐츠의 ID 찾기
        
        Args:
            new_text (str): 새로운 텍스트
            threshold (float): 유사도 임계값
            
        Returns:
            Optional[int]: 매칭된 ID (없으면 None)
        """
        # 제목 제거 (새 파이프라인에서는 "Title: xxx" 형태)
        if new_text.startswith("Title:"):
            lines = new_text.split('\n', 1)
            if len(lines) > 1:
                new_text = lines[1].strip()
        
        normalized_new = self._normalize_text(new_text)
        new_signature = self._create_text_signature(normalized_new)
        
        best_match_id = None
        best_similarity = 0.0
        
        # 1단계: 시그니처 기반 빠른 검색
        for signature, item_id in self.content_to_id_mapping.items():
            similarity = self._calculate_similarity(new_signature, signature)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match_id = item_id
        
        # 2단계: 상위 후보들과 전체 텍스트 비교
        if best_match_id is not None:
            full_text = self.id_to_content_mapping[best_match_id]
            full_similarity = self._calculate_similarity(normalized_new, full_text)
            
            # 전체 텍스트 유사도도 임계값을 넘어야 함
            if full_similarity >= threshold * 0.8:  # 조금 더 관대한 임계값
                return best_match_id
        
        return None
    
    def map_pipeline_results(self, pipeline_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        새 파이프라인 결과를 기존 ground truth ID와 매핑
        
        Args:
            pipeline_results (List[Dict[str, Any]]): 새 파이프라인 결과
            
        Returns:
            List[Dict[str, Any]]: 매핑된 결과 (기존 GT ID 사용)
        """
        mapped_results = []
        successful_mappings = 0
        failed_mappings = []
        
        for i, item in enumerate(pipeline_results):
            original_id = item.get('id', i)
            article_url = item.get('article_url', '')
            original_text = item.get('original_text', '')
            article_title = item.get('article_title', '')
            
            mapped_id = None
            mapping_method = None
            
            # 방법 1: URL 직접 매핑
            if article_url and article_url in self.url_to_gt_mapping:
                mapped_id = self.url_to_gt_mapping[article_url]
                mapping_method = "direct_url"
            
            # 방법 2: 텍스트에서 URL 추출 후 매핑
            elif not mapped_id:
                extracted_url = self.extract_url_from_content(original_text)
                if extracted_url and extracted_url in self.url_to_gt_mapping:
                    mapped_id = self.url_to_gt_mapping[extracted_url]
                    mapping_method = "extracted_url"
            
            # 방법 3: 콘텐츠 유사도 매핑 (가장 강력한 백업)
            if not mapped_id:
                content_match_id = self.find_best_content_match(original_text, threshold=0.5)
                if content_match_id is not None:
                    mapped_id = content_match_id
                    mapping_method = "content_similarity"
            
            # 매핑 결과 처리
            if mapped_id is not None:
                # 성공적인 매핑
                mapped_item = item.copy()
                mapped_item['id'] = mapped_id
                mapped_item['original_pipeline_id'] = original_id
                mapped_item['mapping_method'] = mapping_method
                mapped_results.append(mapped_item)
                successful_mappings += 1
            else:
                # 매핑 실패
                failed_mappings.append({
                    'original_id': original_id,
                    'article_title': article_title,
                    'article_url': article_url,
                    'reason': 'no_matching_ground_truth'
                })
        
        print(f"[INFO] 매핑 결과:")
        print(f"  - 성공: {successful_mappings}개")
        print(f"  - 실패: {len(failed_mappings)}개")
        
        if failed_mappings:
            print("[WARNING] 매핑 실패한 항목들:")
            for failure in failed_mappings[:5]:  # 처음 5개만 표시
                print(f"  - ID {failure['original_id']}: {failure['article_title'][:50]}...")
        
        return mapped_results
    
    def save_mapping_report(self, output_path: str, pipeline_results: List[Dict[str, Any]], mapped_results: List[Dict[str, Any]]):
        """
        매핑 리포트 저장
        
        Args:
            output_path (str): 리포트 저장 경로
            pipeline_results (List[Dict[str, Any]]): 원본 파이프라인 결과
            mapped_results (List[Dict[str, Any]]): 매핑된 결과
        """
        report = {
            "mapping_summary": {
                "total_pipeline_items": len(pipeline_results),
                "successfully_mapped": len(mapped_results),
                "failed_mappings": len(pipeline_results) - len(mapped_results),
                "mapping_rate": len(mapped_results) / len(pipeline_results) if pipeline_results else 0
            },
            "mapping_methods": {},
            "url_mappings_count": len(self.url_to_gt_mapping),
            "content_similarity_mappings_count": len(self.content_to_id_mapping)
        }
        
        # 매핑 방법별 통계
        for item in mapped_results:
            method = item.get('mapping_method', 'unknown')
            report["mapping_methods"][method] = report["mapping_methods"].get(method, 0) + 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 매핑 리포트가 저장되었습니다: {output_path}")


def create_compatible_results(
    pipeline_results_path: str,
    ground_truth_path: str,
    existing_llm_results_path: str,
    output_path: str
) -> str:
    """
    새 파이프라인 결과를 기존 ground truth와 호환되도록 변환
    
    Args:
        pipeline_results_path (str): 새 파이프라인 결과 파일 경로
        ground_truth_path (str): Ground truth 파일 경로  
        existing_llm_results_path (str): 기존 LLM 결과 파일 경로
        output_path (str): 변환된 결과 저장 경로
        
    Returns:
        str: 매핑 리포트 파일 경로
    """
    # ID 매퍼 초기화
    mapper = IDMapper()
    
    # 기존 데이터에서 URL 매핑 구축
    mapper.build_gt_url_mapping(ground_truth_path, existing_llm_results_path)
    mapper.create_content_similarity_mapping(ground_truth_path, existing_llm_results_path)
    
    # 새 파이프라인 결과 로드
    with open(pipeline_results_path, 'r', encoding='utf-8') as f:
        pipeline_results = json.load(f)
    
    # 매핑 수행
    mapped_results = mapper.map_pipeline_results(pipeline_results)
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_results, f, ensure_ascii=False, indent=2)
    
    # 리포트 저장
    report_path = output_path.replace('.json', '_mapping_report.json')
    mapper.save_mapping_report(report_path, pipeline_results, mapped_results)
    
    print(f"[INFO] 호환 가능한 결과가 저장되었습니다: {output_path}")
    return report_path


if __name__ == "__main__":
    # 예시 사용법
    pipeline_results = "/Users/seungho/projects/halluvision/final_pipeline/llm_results/gpt_results_combined.json"
    ground_truth = "/Users/seungho/projects/halluvision/final_pipeline/datasets/ground_truth_ner.json"
    existing_llm = "/Users/seungho/projects/halluvision/final_pipeline/datasets/llm_results_gpt4o_mini_2503.json"
    output_file = "/Users/seungho/projects/halluvision/final_pipeline/llm_results/gpt_results_mapped.json"
    
    create_compatible_results(pipeline_results, ground_truth, existing_llm, output_file)
