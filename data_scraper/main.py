"""
Main Data Scraper module
통합된 데이터 스크래핑 인터페이스 제공
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from data_scraper.crawler import ThreatpostCrawler, MalwarebyteCrawler
from utils.helpers import malware_parser, paragraph_to_content


class DataScraper:
    """
    통합 데이터 스크래퍼 클래스
    다양한 소스로부터 악성코드 정보를 수집하고 통합된 형식으로 제공
    """
    
    def __init__(self, output_dir: str = "scraped_data"):
        """
        Initialize DataScraper
        
        Args:
            output_dir (str): Directory to save scraped data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize crawlers
        # self.wiki_crawler = WikiCrawler()
        self.threatpost_crawler = ThreatpostCrawler()
        self.malwarebyte_crawler = MalwarebyteCrawler()
    
    def scrape_general_data(
        self,
        sources: List[str] = ["threatpost"],
        threatpost_urls_file: Optional[str] = None,
        max_articles: int = 2
    ) -> List[Dict[str, Any]]:
        """
        URL 파일 기반으로 일반적인 보안 기사들을 수집
        
        Args:
            sources (List[str]): 데이터 소스 리스트 ("threatpost", "malwarebyte") 
            threatpost_urls_file (Optional[str]): Threatpost URL 파일 경로
            max_articles (int): 최대 수집할 기사 수
            
        Returns:
            List[Dict[str, Any]]: 수집된 데이터 (각 기사별로 분리)
        """
        print("---------- Processing General Security Articles ----------")
        
        results = []
        
        # Threatpost 크롤링 (URL 파일 기반)
        if "threatpost" in sources:
            try:
                # URL 파일 경로 설정 - datasets 폴더에서 찾기
                if threatpost_urls_file is None:
                    # 기본 경로: datasets 디렉토리의 threatpost_urls.txt
                    base_dir = Path(__file__).parent.parent  # data_scraper의 상위 디렉토리
                    threatpost_urls_file = base_dir / "datasets" / "threatpost_urls.txt"
                
                # URL 파일에서 URL 목록 읽기
                if os.path.exists(threatpost_urls_file):
                    with open(threatpost_urls_file, 'r', encoding='utf-8') as f:
                        urls = [line.strip() for line in f if line.strip()]
                    
                    print(f"Found {len(urls)} Threatpost URLs in {threatpost_urls_file}")
                    
                    # URL들을 크롤링 (제한된 수만)
                    selected_urls = urls[:max_articles]
                    
                    print(f"[INFO] Crawling first {len(selected_urls)} URLs for articles...")
                    threatpost_results = self.threatpost_crawler.crawl_threatpost_by_url(selected_urls)
                    
                    # 각 기사를 개별 항목으로 변환 (LLM 프록시 호환 형식)
                    for i, article in enumerate(threatpost_results):
                        if isinstance(article, dict) and "content" in article:
                            # 제목과 내용을 합쳐서 하나의 텍스트로 만들기
                            combined_content = f"Title: {article.get('title', 'No title')}\n\n{article.get('content', '')}"
                            
                            article_data = {
                                "malware_name": f"threatpost_article_{i+1}",  # LLM 호환성을 위한 식별자
                                "article_title": article.get('title', 'No title'),
                                "article_url": article.get('url', ''),
                                "combined_content": combined_content,
                                "sources": {
                                    "threatpost": {
                                        "url": article.get('url', ''),
                                        "title": article.get('title', ''),
                                        "content_length": len(article.get('content', ''))
                                    }
                                },
                                "metadata": {
                                    "total_paragraphs": 1,
                                    "sources_used": ["threatpost"],
                                    "article_index": i+1,
                                    "source": "threatpost"
                                }
                            }
                            results.append(article_data)
                        
                        elif isinstance(article, str):
                            # 기존 문자열 형태 지원
                            article_data = {
                                "malware_name": f"threatpost_article_{i+1}",
                                "article_title": "Unknown title",
                                "article_url": "",
                                "combined_content": article,
                                "sources": {
                                    "threatpost": {
                                        "content_length": len(article)
                                    }
                                },
                                "metadata": {
                                    "total_paragraphs": 1,
                                    "sources_used": ["threatpost"],
                                    "article_index": i+1,
                                    "source": "threatpost"
                                }
                            }
                            results.append(article_data)
                    
                    print(f"Threatpost: {len(threatpost_results)} articles processed into {len(results)} items")
                else:
                    print(f"Threatpost URLs file not found: {threatpost_urls_file}")
                    
            except Exception as e:
                print(f"Threatpost crawling failed: {e}")
        
        return results

    def scrape_malware_data(
        self, 
        malware_list: List[str], 
        sources: List[str] = ["threatpost"],
        wiki_paragraph_level: int = 3,
        threatpost_urls_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        악성코드 리스트에 대해 지정된 소스들로부터 데이터 수집
        (레거시 지원 - threatpost는 이제 일반 기사 수집 방식)
        
        Args:
            malware_list (List[str]): 수집할 악성코드 이름 리스트
            sources (List[str]): 데이터 소스 리스트 ("wikipedia", "threatpost", "malwarebyte")
            wiki_paragraph_level (int): Wikipedia에서 추출할 단락 수
            threatpost_urls_file (Optional[str]): Threatpost URL 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 수집된 데이터 리스트
        """
        # threatpost가 포함된 경우 일반 기사 수집 방식으로 처리
        if "threatpost" in sources:
            print("[DEPRECATED] Threatpost now uses URL-based general article collection. Using scrape_general_data() method instead...")
            print("=" * 50)
            
            general_data = self.scrape_general_data(
                sources=sources,
                threatpost_urls_file=threatpost_urls_file,
                max_articles=20
            )
            
            # 이미 리스트 형태로 반환되므로 그대로 리턴
            return general_data
        
        # 기존 악성코드별 처리 (wikipedia, malwarebyte 등)
        results = []
        
        for malware_name in malware_list:
            print(f"---------- Processing {malware_name} ----------")
            
            malware_data = {
                "malware_name": malware_name,
                "sources": {},
                "combined_content": "",
                "metadata": {
                    "total_paragraphs": 0,
                    "sources_used": sources.copy()
                }
            }
            
            all_paragraphs = []
            
            # Malwarebytes 크롤링
            if "malwarebyte" in sources:
                try:
                    mb_results = self.malwarebyte_crawler.crawl_malwarebyte(malware_name)
                    
                    malware_data["sources"]["malwarebyte"] = {
                        "articles": mb_results,
                        "article_count": len(mb_results)
                    }
                    all_paragraphs.extend(mb_results)
                    print(f"Malwarebytes: {len(mb_results)} articles")
                    
                except Exception as e:
                    print(f"Malwarebytes crawling failed for {malware_name}: {e}")
                    malware_data["sources"]["malwarebyte"] = {"error": str(e)}
            
            # 통합 콘텐츠 생성
            malware_data["combined_content"] = paragraph_to_content(all_paragraphs)
            malware_data["metadata"]["total_paragraphs"] = len(all_paragraphs)
            
            results.append(malware_data)
            
            # 중간 저장
            self._save_intermediate_result(malware_data)
        
        return results
    
    def scrape_from_file(
        self, 
        malware_file_path: str, 
        sources: List[str] = ["threatpost"],
        wiki_paragraph_level: int = 3,
        threatpost_urls_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        파일에서 악성코드 리스트를 읽어와 데이터 수집
        
        Args:
            malware_file_path (str): 악성코드 리스트 파일 경로
            sources (List[str]): 데이터 소스 리스트
            wiki_paragraph_level (int): Wikipedia에서 추출할 단락 수
            threatpost_urls_file (Optional[str]): Threatpost URL 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 수집된 데이터 리스트
        """
        malware_list = malware_parser(malware_file_path)
        return self.scrape_malware_data(malware_list, sources, wiki_paragraph_level, threatpost_urls_file)
    
    def save_results(self, results: List[Dict[str, Any]], filepath: str = "scraped_data.json"):
        """
        수집된 결과를 JSON 파일로 저장
        
        Args:
            results (List[Dict[str, Any]]): 저장할 데이터
            filepath (str): 저장할 파일 경로 (절대경로 또는 파일명)
        """
        # 절대경로인지 확인하고, 그렇지 않으면 output_dir 기준으로 처리
        if os.path.isabs(filepath):
            output_path = Path(filepath)
        else:
            output_path = self.output_dir / filepath
        
        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def _save_intermediate_result(self, malware_data: Dict[str, Any]):
        """중간 결과 저장 (개별 악성코드별)"""
        filename = f"{malware_data['malware_name']}_data.json"
        filepath = self.output_dir / "intermediate" / filename
        
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(malware_data, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        수집된 데이터의 통계 정보 반환
        
        Args:
            results (List[Dict[str, Any]]): 분석할 데이터
            
        Returns:
            Dict[str, Any]: 통계 정보
        """
        stats = {
            "total_malware": len(results),
            "total_paragraphs": 0,
            "source_stats": {},
            "malware_stats": {}
        }
        
        for result in results:
            malware_name = result["malware_name"]
            total_paras = result["metadata"]["total_paragraphs"]
            
            stats["total_paragraphs"] += total_paras
            stats["malware_stats"][malware_name] = total_paras
            
            for source, data in result["sources"].items():
                if source not in stats["source_stats"]:
                    stats["source_stats"][source] = {"count": 0, "total_items": 0}
                
                stats["source_stats"][source]["count"] += 1
                
                if "paragraph_count" in data:
                    stats["source_stats"][source]["total_items"] += data["paragraph_count"]
                elif "article_count" in data:
                    stats["source_stats"][source]["total_items"] += data["article_count"]
        
        return stats


# Legacy function for backward compatibility
def malware_crawler(kb_type: str, malware_src: str, wiki_paragraph_level: int = 2):
    """
    Legacy wrapper function for existing malware_crawler calls
    
    Args:
        kb_type (str): Source type ("wikipedia", "threatpost", "malwarebyte")
        malware_src (str): Path to malware list file
        wiki_paragraph_level (int): Number of paragraphs to extract
    """
    scraper = DataScraper()
    
    # Map legacy kb_type to new sources format
    sources = [kb_type] if kb_type in ["wikipedia", "threatpost", "malwarebyte"] else ["wikipedia"]
    
    results = scraper.scrape_from_file(malware_src, sources, wiki_paragraph_level)
    
    # Save results for backward compatibility
    scraper.save_results(results, f"{kb_type}_crawled_data.json")
    
    # Print legacy-style statistics
    stats = scraper.get_statistics(results)
    print(f'---------- ARTICLES INFO ({kb_type}) [ASCENDING] ----------')
    
    sorted_malware_stats = dict(sorted(stats["malware_stats"].items(), key=lambda item: item[1]))
    for malware, count in sorted_malware_stats.items():
        print(f'{malware}: {count}', end=' ')
    print(f'total Articles : {stats["total_paragraphs"]}')


if __name__ == "__main__":
    # Example usage
    scraper = DataScraper()
    
    # Test with a small list
    test_malware = ["zeus", "stuxnet"]
    results = scraper.scrape_malware_data(test_malware, sources=["wikipedia"])
    
    # Save results
    scraper.save_results(results, "test_scraped_data.json")
    
    # Print statistics
    stats = scraper.get_statistics(results)
    print("\n--- Scraping Statistics ---")
    print(f"Total malware processed: {stats['total_malware']}")
    print(f"Total paragraphs collected: {stats['total_paragraphs']}")
    print("Source statistics:", stats['source_stats'])
