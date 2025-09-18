"""
Helper utilities for data processing
"""

def malware_parser(file_path):
    """
    Parse malware list from file
    
    Args:
        file_path (str): Path to the malware list file
        
    Returns:
        list: List of malware names
    """
    malware_lists = []
    with open(file_path, "r") as file:
        malware_lists = file.read().splitlines()
    return malware_lists


def paragraph_to_content(paragraphs):
    """
    Convert list of paragraphs to single content string
    
    Args:
        paragraphs (list): List of paragraph strings
        
    Returns:
        str: Combined content string
    """
    return " ".join(paragraphs)


def list_contents_words_len(contents):
    """
    Count total words in list of contents
    
    Args:
        contents (list): List of content strings
        
    Returns:
        int: Total word count
    """
    cnt = 0
    for para in contents:
        cnt += len(para)
    return cnt
