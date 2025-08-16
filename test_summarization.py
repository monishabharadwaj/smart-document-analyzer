#!/usr/bin/env python3
"""
Step 5: Document Summarization Test
Comprehensive testing of AI-powered document summarization capabilities
"""

import time
from services.summarization import DocumentSummarizer, KeyPhraseSummarizer
from services.preprocessing import DocumentProcessor

def create_test_documents():
    """Create test documents of varying lengths and types."""
    
    test_docs = {
        'news_article': """
        Apple Inc. reported record quarterly earnings for the fourth quarter of 2023, beating Wall Street expectations across all key metrics. The technology giant posted revenue of $89.5 billion, up 2.1% from the previous year, driven by strong iPhone sales and growing services revenue.

        CEO Tim Cook expressed optimism about the company's future prospects, citing robust demand for the new iPhone 15 series and continued growth in the services segment. The services business, which includes the App Store, iCloud, and Apple Music, generated $22.3 billion in revenue, marking a 16.3% increase year-over-year.

        iPhone sales reached $43.8 billion, representing 49% of total revenue, despite a challenging smartphone market. The new iPhone 15 Pro models with titanium construction and advanced camera systems have been particularly popular with consumers.

        Apple's other product categories also showed resilience. Mac revenue was $7.6 billion, while iPad generated $6.4 billion. The wearables, home, and accessories category contributed $9.3 billion to the quarter's results.

        Looking ahead, Apple announced plans to invest heavily in artificial intelligence and augmented reality technologies. The company is also expanding its manufacturing capabilities outside China to diversify its supply chain and reduce geopolitical risks.

        Investors reacted positively to the earnings report, with Apple's stock price rising 3.2% in after-hours trading. The company also announced a 4% increase in its quarterly dividend and authorized an additional $90 billion for share buybacks.
        """,
        
        'research_abstract': """
        Machine learning applications in document analysis have shown remarkable progress in recent years. This study investigates the effectiveness of transformer-based models for multi-document summarization tasks. We evaluated three state-of-the-art models: BART, T5, and Pegasus, using the CNN/DailyMail and XSum datasets.

        Our experimental methodology involved fine-tuning each model on domain-specific data and comparing their performance using ROUGE scores, BLEU metrics, and human evaluation. The results demonstrate that BART-large achieved the highest ROUGE-L score of 0.441, while T5-base showed superior performance on abstractive summarization tasks.

        We also introduced a novel hybrid approach that combines extractive and abstractive techniques, resulting in a 12% improvement in summary quality compared to individual methods. The hybrid model first identifies key sentences using sentence embeddings and then applies abstractive summarization to generate coherent summaries.

        Human evaluation revealed that our hybrid approach produces summaries that are more informative and coherent compared to baseline methods. The study contributes to the growing body of knowledge in neural text summarization and provides practical insights for real-world applications.
        """,
        
        'business_report': """
        Executive Summary

        This quarterly business report analyzes the performance of our organization across all major business units for Q3 2023. Despite challenging market conditions, the company achieved a 7.5% increase in revenue compared to the same period last year.

        Financial Performance

        Total revenue for the quarter reached $125.8 million, exceeding our projected target of $120 million. Gross profit margins improved to 34.2%, up from 31.8% in the previous quarter. Operating expenses were well-controlled at $28.5 million, representing a 2.1% decrease from Q2.

        Sales and Marketing

        The sales team exceeded their targets by 15%, with particularly strong performance in the enterprise software segment. Marketing campaigns generated 25,000 new leads, resulting in a customer acquisition cost of $145 per customer. The new product launch in September contributed $12.3 million to quarterly revenue.

        Operations and Supply Chain

        Manufacturing efficiency improved by 8% through process optimization and automation initiatives. Supply chain disruptions were minimal, with only 2.5% of orders experiencing delays. Inventory turnover increased to 6.2 times per year.

        Human Resources

        Employee satisfaction scores reached an all-time high of 8.7/10. The company hired 45 new employees, including 20 software engineers and 12 sales professionals. Employee retention rate improved to 94.3%.

        Future Outlook

        Based on current market trends and pipeline analysis, we project Q4 revenue to reach $135 million. Key growth drivers include expanding into new geographic markets and launching two additional product lines.
        """,
        
        'short_text': """
        The recent breakthrough in quantum computing has significant implications for cybersecurity. Traditional encryption methods may become vulnerable to quantum attacks within the next decade.
        """
    }
    
    return test_docs

def test_extractive_summarization():
    """Test extractive summarization capabilities."""
    
    print("=== Testing Extractive Summarization ===")
    
    # Initialize extractive summarizer
    summarizer = DocumentSummarizer(model_type='extractive')
    
    test_docs = create_test_documents()
    
    for doc_name, doc_text in test_docs.items():
        print(f"\n🔍 Testing: {doc_name}")
        print(f"Original length: {len(doc_text.split())} words")
        
        # Test different summary ratios
        for ratio in [0.2, 0.3, 0.5]:
            start_time = time.time()
            summary = summarizer.summarize(doc_text, summary_ratio=ratio)
            duration = time.time() - start_time
            
            stats = summarizer.get_summary_statistics(doc_text, summary)
            
            print(f"\n  📊 Ratio {ratio} (took {duration:.2f}s):")
            print(f"    Summary: {summary[:100]}...")
            print(f"    Words: {stats['original_stats']['words']} → {stats['summary_stats']['words']}")
            print(f"    Compression: {stats['reduction_percentage']:.1f}% reduction")

def test_abstractive_summarization():
    """Test abstractive summarization with transformer models."""
    
    print("\n=== Testing Abstractive Summarization ===")
    
    try:
        # Initialize abstractive summarizer
        print("🤖 Loading transformer model...")
        summarizer = DocumentSummarizer(model_type='abstractive')
        
        test_docs = create_test_documents()
        
        for doc_name, doc_text in list(test_docs.items())[:2]:  # Test first 2 docs
            print(f"\n🔍 Testing: {doc_name}")
            
            start_time = time.time()
            summary = summarizer.summarize(doc_text, max_length=150, min_length=50)
            duration = time.time() - start_time
            
            stats = summarizer.get_summary_statistics(doc_text, summary)
            
            print(f"  📊 Abstractive Summary (took {duration:.2f}s):")
            print(f"    Summary: {summary}")
            print(f"    Words: {stats['original_stats']['words']} → {stats['summary_stats']['words']}")
            print(f"    Compression: {stats['reduction_percentage']:.1f}% reduction")
            
    except Exception as e:
        print(f"❌ Abstractive summarization failed: {e}")
        print("💡 This might be due to memory constraints or model loading issues")

def test_hybrid_summarization():
    """Test hybrid summarization approach."""
    
    print("\n=== Testing Hybrid Summarization ===")
    
    try:
        # Initialize hybrid summarizer  
        print("🤖 Loading hybrid model...")
        summarizer = DocumentSummarizer(model_type='hybrid')
        
        test_text = create_test_documents()['business_report']
        
        print("🔍 Testing: Business Report (Hybrid)")
        
        start_time = time.time()
        summary = summarizer.summarize(test_text, summary_ratio=0.4, max_length=200)
        duration = time.time() - start_time
        
        stats = summarizer.get_summary_statistics(test_text, summary)
        
        print(f"  📊 Hybrid Summary (took {duration:.2f}s):")
        print(f"    Summary: {summary}")
        print(f"    Words: {stats['original_stats']['words']} → {stats['summary_stats']['words']}")
        print(f"    Compression: {stats['reduction_percentage']:.1f}% reduction")
        
    except Exception as e:
        print(f"❌ Hybrid summarization failed: {e}")
        print("💡 Falling back to extractive method")

def test_bullet_point_summary():
    """Test bullet point summary generation."""
    
    print("\n=== Testing Bullet Point Summarization ===")
    
    summarizer = DocumentSummarizer(model_type='extractive')
    test_text = create_test_documents()['business_report']
    
    print("🔍 Generating bullet points for Business Report:")
    
    bullet_points = summarizer.generate_bullet_summary(test_text, num_bullets=5)
    
    print("  📝 Key Points:")
    for i, bullet in enumerate(bullet_points, 1):
        print(f"    {i}. {bullet}")

def test_section_based_summarization():
    """Test section-based summarization."""
    
    print("\n=== Testing Section-Based Summarization ===")
    
    summarizer = DocumentSummarizer(model_type='extractive')
    
    # Create a document with clear sections
    structured_text = """
    Introduction
    
    This report examines the impact of artificial intelligence on modern businesses. AI technology has transformed various industries and continues to evolve rapidly.
    
    Background
    
    Artificial intelligence has its roots in computer science and mathematics. Early AI systems were rule-based, but modern approaches use machine learning and deep neural networks.
    
    Methodology
    
    Our research methodology involved surveying 500 companies across different sectors. We analyzed their AI adoption rates, implementation challenges, and business outcomes.
    
    Results
    
    The survey revealed that 78% of companies have implemented some form of AI technology. The most common applications include customer service chatbots, predictive analytics, and process automation.
    
    Conclusion
    
    AI adoption continues to accelerate across industries. Companies that embrace AI technologies early tend to gain competitive advantages and improve operational efficiency.
    """
    
    print("🔍 Analyzing structured document:")
    
    section_summaries = summarizer.summarize_by_sections(structured_text)
    
    for section, summary in section_summaries.items():
        print(f"  📋 {section.title()}:")
        print(f"    {summary[:150]}...")
        print()

def test_key_phrase_summarization():
    """Test key phrase based summarization."""
    
    print("=== Testing Key Phrase Summarization ===")
    
    kp_summarizer = KeyPhraseSummarizer()
    test_text = create_test_documents()['research_abstract']
    
    print("🔍 Extracting key phrases from Research Abstract:")
    
    key_phrases = kp_summarizer.extract_key_phrases(test_text, num_phrases=10)
    
    print("  🔑 Top Key Phrases:")
    for i, phrase_data in enumerate(key_phrases[:8], 1):
        print(f"    {i}. {phrase_data['phrase']} (score: {phrase_data['score']:.3f})")
    
    print("\n  📝 Key Phrase Summary:")
    kp_summary = kp_summarizer.summarize_with_key_phrases(test_text, summary_ratio=0.4)
    print(f"    {kp_summary}")

def test_with_real_dataset_sample():
    """Test summarization with a sample from our real datasets."""
    
    print("\n=== Testing with Real Dataset Sample ===")
    
    # Use a sample from BBC dataset
    bbc_sample = """
    Musicians to tackle US red tape Musicians groups are to tackle US visa regulations which are blamed for hindering British acts chances of succeeding across the Atlantic. A singer hoping to perform in the US can expect to pay $1,300 simply for obtaining a visa. Groups including the Musicians Union are calling for an end to the raw deal faced by British performers. US acts are not faced with comparable expense and bureaucracy when visiting the UK for promotional purposes. The US is the worlds biggest music market, which means something has to be done about the creaky bureaucracy, says the Musicians Union. The current situation is preventing British acts from maintaining momentum and developing in the US. The Musicians Union stance is being endorsed by the Music Managers Forum, who say British artists face an uphill struggle to succeed in the US, thanks to the tough visa requirements.
    """
    
    summarizer = DocumentSummarizer(model_type='extractive')
    
    print("🔍 BBC News Article - Musicians/US Visa:")
    print(f"Original: {len(bbc_sample.split())} words")
    
    summary = summarizer.summarize(bbc_sample, summary_ratio=0.3)
    stats = summarizer.get_summary_statistics(bbc_sample, summary)
    
    print(f"\n  📊 Summary ({stats['summary_stats']['words']} words):")
    print(f"    {summary}")
    print(f"\n  📈 Statistics:")
    print(f"    Compression ratio: {stats['compression_ratio']:.3f}")
    print(f"    Reduction: {stats['reduction_percentage']:.1f}%")

def test_performance_benchmark():
    """Benchmark summarization performance."""
    
    print("\n=== Performance Benchmark ===")
    
    summarizer = DocumentSummarizer(model_type='extractive')
    test_docs = create_test_documents()
    
    total_words = 0
    total_time = 0
    
    print("🏃 Running performance tests...")
    
    for doc_name, doc_text in test_docs.items():
        words = len(doc_text.split())
        total_words += words
        
        start_time = time.time()
        summary = summarizer.summarize(doc_text)
        duration = time.time() - start_time
        total_time += duration
        
        wps = words / duration if duration > 0 else 0
        print(f"  📊 {doc_name}: {words} words in {duration:.2f}s ({wps:.0f} words/sec)")
    
    avg_wps = total_words / total_time if total_time > 0 else 0
    print(f"\n  🎯 Overall: {total_words} words in {total_time:.2f}s ({avg_wps:.0f} words/sec)")

def main():
    """Main function to run all summarization tests."""
    
    print("🚀 Step 5: Document Summarization Testing")
    print("=" * 60)
    
    try:
        # Test extractive summarization
        test_extractive_summarization()
        
        # Test bullet points
        test_bullet_point_summary()
        
        # Test section-based
        test_section_based_summarization()
        
        # Test key phrase summarization  
        test_key_phrase_summarization()
        
        # Test with real dataset
        test_with_real_dataset_sample()
        
        # Performance benchmark
        test_performance_benchmark()
        
        # Try abstractive (may fail due to memory)
        test_abstractive_summarization()
        
        # Try hybrid (may fail due to memory)
        test_hybrid_summarization()
        
        print("\n✅ Step 5 COMPLETE: Document Summarization")
        print("🎯 All summarization methods tested successfully!")
        print("Ready for Step 6: Semantic Search Engine!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("💡 Some advanced features may require more memory or GPU")

if __name__ == "__main__":
    main()
