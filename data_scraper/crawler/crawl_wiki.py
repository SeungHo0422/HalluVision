"""
Wikipedia Crawler for malware information
"""

import requests
import wikipedia
from typing import List, Optional
import time


class WikiCrawler:
    """Wikipedia crawler for malware information"""
    
    def __init__(self):
        wikipedia.set_lang("en")
    
    def crawl_wiki(self, search_query: str, paragraph_level: int = 2) -> List[str]:
        """
        Crawl Wikipedia for malware information
        
        Args:
            search_query (str): Search query for Wikipedia
            paragraph_level (int): Number of paragraphs to extract
            
        Returns:
            List[str]: List of extracted paragraphs
        """
        try:
            # Search for Wikipedia pages
            search_results = wikipedia.search(search_query, results=5)
            
            if not search_results:
                print(f"No Wikipedia results found for: {search_query}")
                return []
            
            # Get the first result
            page_title = search_results[0]
            print(f"Found Wikipedia page: {page_title}")
            
            # Get page content
            page = wikipedia.page(page_title)
            
            # Split content into paragraphs
            paragraphs = page.content.split('\n\n')
            
            # Filter out empty paragraphs and take specified number
            filtered_paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Return specified number of paragraphs
            return filtered_paragraphs[:paragraph_level] if filtered_paragraphs else []
            
        except wikipedia.exceptions.DisambiguationError as e:
            # If there are multiple options, try the first one
            try:
                page = wikipedia.page(e.options[0])
                paragraphs = page.content.split('\n\n')
                filtered_paragraphs = [p.strip() for p in paragraphs if p.strip()]
                return filtered_paragraphs[:paragraph_level] if filtered_paragraphs else []
            except Exception as ex:
                print(f"Error processing disambiguation: {ex}")
                return []
                
        except wikipedia.exceptions.PageError:
            print(f"Wikipedia page not found for: {search_query}")
            return []
            
        except Exception as e:
            print(f"Error crawling Wikipedia: {e}")
            return []


# Legacy function for backward compatibility
def crawl_wiki(search_query: str, paragraph_level: int = 2) -> List[str]:
    """
    Legacy wrapper function for WikiCrawler
    
    Args:
        search_query (str): Search query for Wikipedia
        paragraph_level (int): Number of paragraphs to extract
        
    Returns:
        List[str]: List of extracted paragraphs
    """
    crawler = WikiCrawler()
    return crawler.crawl_wiki(search_query, paragraph_level)


if __name__ == "__main__":
    # Test the crawler
    crawler = WikiCrawler()
    test_query = "Zeus malware wikipedia"
    results = crawler.crawl_wiki(test_query, 3)
    
    print(f"Results for '{test_query}':")
    for i, paragraph in enumerate(results, 1):
        print(f"{i}. {paragraph[:100]}...")
