"""
Threatpost Crawler for malware information
"""

from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By
import time
from typing import List, Optional


class ThreatpostCrawler:
    """Threatpost crawler for malware information"""
    
    def __init__(self, headless: bool = True):
        """
        Initialize Threatpost crawler
        
        Args:
            headless (bool): Whether to run browser in headless mode
        """
        self.headless = headless
        self.driver = None
    
    def start_webdriver_session(self):
        """Start a new webdriver session"""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        return self.driver
    
    def crawl_threatpost_by_url(self, urls: List[str]) -> List[dict]:
        """
        Crawl Threatpost articles by URLs
        
        Args:
            urls (List[str]): List of Threatpost URLs to crawl
            
        Returns:
            List[dict]: List of extracted articles with title and content
        """
        if not self.driver:
            self.start_webdriver_session()
        
        results = []
        success, fail, session_count = 0, 0, 0
        
        for cur_url in urls:
            if session_count > 400:
                # session restart
                session_count = 0
                self.driver.quit()
                self.start_webdriver_session()
            
            try:
                self.driver.get(cur_url)
                self.driver.implicitly_wait(5)
                print(f"Processing: {self.driver.current_url}")
                
                # Extract title and content from the page
                try:
                    # Extract title
                    title = ""
                    try:
                        title_element = self.driver.find_element(By.TAG_NAME, "h1")
                        title = title_element.text.strip()
                    except:
                        # Try alternative title selectors
                        try:
                            title_element = self.driver.find_element(By.CLASS_NAME, "c-article__title")
                            title = title_element.text.strip()
                        except:
                            title = "No title found"
                    
                    # Extract content
                    content = ""
                    search_results_paragraphs = self.driver.find_element(
                        By.CLASS_NAME, "c-article__main"
                    ).find_elements(By.TAG_NAME, "p")
                    
                    self.driver.implicitly_wait(5)
                    for para in search_results_paragraphs:
                        content += para.text + " "
                    
                    # Create article object
                    article = {
                        "url": cur_url,
                        "title": title,
                        "content": content.strip(),
                        "source": "threatpost"
                    }
                    
                    results.append(article)
                    success += 1
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f'[ERROR] article error: {e}')
                    fail += 1
                    continue
                    
                session_count += 1
                
            except WebDriverException as e:
                print(f'[ERROR] WebDriverException: {e}')
                continue
            except Exception as e:
                print(f'[ERROR] ({cur_url}): {e}')
                continue
        
        if self.driver:
            self.driver.quit()
        
        print(f'Total stats: SUCCESS {success}, FAIL {fail}')
        return results
    
    def crawl_threatpost_category(self, category_url: str = "https://threatpost.com/category/malware-2/") -> List[str]:
        """
        Crawl Threatpost malware category page
        
        Args:
            category_url (str): URL of the category page
            
        Returns:
            List[str]: List of article URLs
        """
        if not self.driver:
            self.start_webdriver_session()
        
        article_urls = []
        
        try:
            self.driver.get(category_url)
            self.driver.implicitly_wait(3)
            print(f'URL: {self.driver.current_url}')
            
            # Accept cookies
            try:
                self.driver.find_element(By.CLASS_NAME, "gdprButton").click()
                self.driver.implicitly_wait(3)
            except:
                pass  # Cookie button might not be present
            
            # Load more articles
            next_button = self.driver.find_element(By.ID, "load_more_archive")
            
            try:
                for i in range(100):  # Limit iterations
                    next_button.click()
                    self.driver.implicitly_wait(5)
                    time.sleep(2)
            except Exception as e:
                print(f'[INFO] Finished loading articles after {i+1} clicks: {e}')
            
            # Extract article URLs
            search_results = self.driver.find_elements(By.CLASS_NAME, "c-card__title")
            self.driver.implicitly_wait(10)
            
            for result in search_results:
                try:
                    cur_url = result.find_element(By.TAG_NAME, "a").get_attribute("href")
                    article_urls.append(cur_url)
                except:
                    continue
            
        except Exception as e:
            print(f'[ERROR] Failed to crawl category page: {e}')
        
        finally:
            if self.driver:
                self.driver.quit()
        
        return article_urls


# Legacy functions for backward compatibility
def crawl_threatpost_by_url(urls):
    """Legacy wrapper function"""
    crawler = ThreatpostCrawler()
    return crawler.crawl_threatpost_by_url(urls)


def crawl_threatpost(category_url="https://threatpost.com/category/malware-2/"):
    """Legacy wrapper function"""
    crawler = ThreatpostCrawler()
    return crawler.crawl_threatpost_category(category_url)


if __name__ == "__main__":
    # Test the crawler
    crawler = ThreatpostCrawler()
    
    # Test with a single URL (replace with actual Threatpost URL)
    test_urls = ["https://threatpost.com/sample-article-url"]
    results = crawler.crawl_threatpost_by_url(test_urls)
    
    print(f"Crawled {len(results)} articles")
    for i, article in enumerate(results[:2], 1):
        print(f"{i}. {article[:200]}...")
