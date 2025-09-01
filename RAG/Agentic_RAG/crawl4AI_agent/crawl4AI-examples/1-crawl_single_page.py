import asyncio
from crawl4ai import *

async def main():
    print("Starting web crawler...")
    try:
        async with AsyncWebCrawler() as crawler:
            print("Crawling https://ai.pydantic.dev/...") # Crawl the UI, becomes slow for large websites. As it opens every link.
            result = await crawler.arun(
                url="https://ai.pydantic.dev/",
            )
            print("Crawling completed!")
            print("Content length:", len(result.markdown) if result.markdown else 0)
            print("\n" + "="*50)
            print("COMPLETE DOCUMENT:")
            print("="*50)
            print(result.markdown)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())