#!/usr/bin/env python3
"""
Test script for Named Entity Recognition (NER) service
"""

from services.ner import NERService

def test_ner():
    print("=== Step 3: Named Entity Recognition (NER) Test ===")
    
    # Initialize NER service
    ner_service = NERService()
    
    # Read sample document
    with open('sample_document.txt', 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print("\n🔍 Extracting entities from sample document...")
    entities = ner_service.extract_entities(sample_text)
    
    print("\n📊 Standard Entities Found:")
    for entity_type, entity_list in entities.items():
        print(f"  {entity_type}: {len(entity_list)} entities")
        for entity in entity_list[:2]:  # Show first 2 entities of each type
            text = entity['text']
            start = entity['start']
            end = entity['end']
            print(f"    - '{text}' (position: {start}-{end})")
        if len(entity_list) > 2:
            print(f"    ... and {len(entity_list)-2} more")
    
    print("\n💰 Financial Entities:")
    financial_entities = ner_service.extract_financial_entities(sample_text)
    for entity_type, entity_list in financial_entities.items():
        entity_texts = [e['text'] for e in entity_list]
        print(f"  {entity_type}: {entity_texts}")
    
    print("\n📋 Document Metadata Entities:")
    metadata_entities = ner_service.extract_document_metadata(sample_text)
    for entity_type, entity_list in metadata_entities.items():
        entity_texts = [e['text'] for e in entity_list]
        print(f"  {entity_type}: {entity_texts}")
    
    print("\n📈 Entity Summary:")
    summary = ner_service.get_entity_summary(sample_text)
    print(f"  Total entities found: {summary['total_entities']}")
    print(f"  Entity types: {summary['entity_types']}")
    print(f"  Entities by type: {summary['entities_by_type']}")
    
    print("\n✅ Step 3 COMPLETE: Named Entity Recognition")
    print("Ready for Step 4: Document Classification!")
    
    return True

if __name__ == "__main__":
    test_ner()
