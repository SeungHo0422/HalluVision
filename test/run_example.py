#!/usr/bin/env python3
"""
Example script to run the HalluVision Pipeline with Threatpost URLs
Threatpost URL을 이용한 파이프라인 실행 예제
"""

import os
from pathlib import Path
from main import HallucinationVisionPipeline

def main():
    """메인 실행 함수"""
    
    # 경로 설정
    current_dir = Path(__file__).parent
    malware_file = current_dir / "example_malware_list.txt"
    threatpost_urls_file = current_dir / "data_scraper" / "threatpost_urls.txt"
    output_dir = current_dir / "example_output"
    
    print("🔍 Threatpost URL 기반 파이프라인 실행 예제")
    print("=" * 60)
    
    # 파일 존재 확인
    print("파일 확인 중...")
    if not malware_file.exists():
        print(f"❌ Malware 리스트 파일을 찾을 수 없습니다: {malware_file}")
        return
    
    if not threatpost_urls_file.exists():
        print(f"❌ Threatpost URL 파일을 찾을 수 없습니다: {threatpost_urls_file}")
        return
    
    print(f"✅ Malware 리스트: {malware_file}")
    print(f"✅ Threatpost URLs: {threatpost_urls_file}")
    print(f"📁 출력 디렉토리: {output_dir}")
    
    # API 키 확인
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OpenAI API key가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key'")
        print("   현재는 LLM 처리 없이 데이터 스크래핑만 실행합니다.")
        tasks = []  # LLM 태스크 없음
    else:
        print("✅ OpenAI API key 확인됨")
        tasks = ["summarization", "qa"]
    
    print()
    
    try:
        # 파이프라인 초기화
        pipeline = HallucinationVisionPipeline(str(output_dir))
        
        # 파이프라인 실행
        print("🚀 파이프라인 시작...")
        
        # 데이터 스크래핑만 먼저 실행
        print("\n📊 1단계: Threatpost 데이터 스크래핑")
        scraped_data = pipeline.run_data_scraping(
            malware_file_path=str(malware_file),
            sources=["threatpost"],
            threatpost_urls_file=str(threatpost_urls_file)
        )
        
        print(f"✅ 스크래핑 완료: {len(scraped_data)}개 항목")
        
        # LLM 처리 (API 키가 있는 경우에만)
        if tasks:
            print("\n🤖 2단계: LLM 처리")
            llm_results = pipeline.run_llm_processing(tasks)
            print(f"✅ LLM 처리 완료: {len(llm_results)}개 태스크")
        
        print("\n🎉 예제 실행 완료!")
        print(f"📁 결과는 {output_dir}에 저장되었습니다.")
        
        # 결과 요약 출력
        print("\n📈 결과 요약:")
        for data in scraped_data[:3]:  # 처음 3개만 출력
            malware_name = data.get("malware_name", "Unknown")
            sources_used = data.get("metadata", {}).get("sources_used", [])
            content_length = len(data.get("combined_content", ""))
            
            print(f"  - {malware_name}: {sources_used} ({content_length} characters)")
        
        if len(scraped_data) > 3:
            print(f"  ... 총 {len(scraped_data)}개 항목")
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
