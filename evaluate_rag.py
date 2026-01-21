# evaluate_rag.py
"""
MediQuery RAG Comprehensive Evaluation
Tests different query types and categories
"""

import time
from backend.rag.chain import get_rag_chain


# 9 comprehensive test queries covering different aspects
EVAL_QUERIES = [
    # 1. Simple side effects (easy)
    {
        "question": "What are the side effects of Adderall?",
        "keywords": ["cardiovascular", "psychiatric", "appetite"],
        "category": "side_effects",
        "difficulty": "easy"
    },
    
    # 2. Pregnancy/contraindication (critical safety)
    {
        "question": "Can pregnant women take Accutane?",
        "keywords": ["contraindicated", "pregnancy", "birth defects"],
        "category": "pregnancy",
        "difficulty": "easy"
    },
    
    # 3. Indications/uses (basic info)
    {
        "question": "What is Lipitor used for?",
        "keywords": ["cholesterol", "cardiovascular"],
        "category": "indications",
        "difficulty": "easy"
    },
    
    # 4. Drug interactions (medium complexity)
    {
        "question": "What are drug interactions with Ibuprofen?",
        "keywords": ["bleeding", "anticoagulant"],
        "category": "interactions",
        "difficulty": "medium"
    },
    
    # 5. Warnings/precautions (important safety)
    {
        "question": "What warnings exist for Metformin?",
        "keywords": ["lactic acidosis", "renal"],
        "category": "warnings",
        "difficulty": "medium"
    },
    
    # 6. Dosage information (specific details)
    {
        "question": "What is the recommended dosage for Prozac in adults?",
        "keywords": ["mg", "daily", "dose"],
        "category": "dosage",
        "difficulty": "medium"
    },
    
    # 7. Contraindications (when NOT to use)
    {
        "question": "When should Lisinopril not be used?",
        "keywords": ["contraindicated", "pregnancy", "angioedema"],
        "category": "contraindications",
        "difficulty": "medium"
    },
    
    # 8. Comparison question (harder)
    {
        "question": "How do the side effects of Ritalin compare to Adderall?",
        "keywords": ["stimulant", "adhd", "cardiovascular"],
        "category": "comparison",
        "difficulty": "hard"
    },
    
    # 9. Multi-part question (complex)
    {
        "question": "What should I know about taking Tretinoin - its uses, side effects, and precautions?",
        "keywords": ["acne", "skin", "photosensitivity", "pregnancy"],
        "category": "multi_part",
        "difficulty": "hard"
    }
]


def main():
    print("MediQuery Comprehensive RAG Evaluation")
    print(f"Testing {len(EVAL_QUERIES)} queries across different categories")
    
    chain = get_rag_chain()
    results = []
    
    # Track by category
    by_category = {}
    by_difficulty = {}
    
    # Run queries
    for i, q in enumerate(EVAL_QUERIES, 1):
        cat = q['category']
        diff = q['difficulty']
        
        print(f"\n[{i}/{len(EVAL_QUERIES)}] {q['question']}")
        print(f"  Category: {cat} | Difficulty: {diff}")
        
        start = time.time()
        answer = chain.invoke(q['question'], verbose=False)
        latency_s = time.time() - start
        
        # Check coverage
        answer_lower = answer.lower()
        found = [kw for kw in q['keywords'] if kw in answer_lower]
        coverage = len(found) / len(q['keywords'])
        
        # Quality checks
        has_source = any(x in answer for x in ["Source", ".pdf", "p."])
        length = len(answer)
        success = coverage >= 0.4 and length >= 200
        
        # Store result
        result = {
            'success': success,
            'coverage': coverage,
            'latency_s': latency_s,
            'has_source': has_source,
            'length': length,
            'category': cat,
            'difficulty': diff,
            'found': found,
            'total': len(q['keywords'])
        }
        results.append(result)
        
        # Track by category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)
        
        # Track by difficulty
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(result)
        
        # Print result
        status = "✅" if success else "❌"
        print(f"  {status} {latency_s:.1f}s | Coverage: {coverage:.0%} ({len(found)}/{len(q['keywords'])}) | {length} chars")
        if found:
            print(f"     Found: {', '.join(found)}")
    
    # Overall Summary
    print("OVERALL SUMMARY")
    
    passed = sum(1 for r in results if r['success'])
    avg_time = sum(r['latency_s'] for r in results) / len(results)
    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    avg_length = sum(r['length'] for r in results) / len(results)
    with_source = sum(1 for r in results if r['has_source'])
    
    print(f"\nSuccess Rate:     {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")
    print(f"Avg Response:     {avg_time:.1f}s")
    print(f"Avg Coverage:     {avg_coverage:.0%}")
    print(f"Avg Length:       {avg_length:.0f} chars")
    print(f"With Sources:     {with_source}/{len(results)}")
    
    print("BY CATEGORY")
    
    for cat in sorted(by_category.keys()):
        cat_results = by_category[cat]
        cat_passed = sum(1 for r in cat_results if r['success'])
        cat_coverage = sum(r['coverage'] for r in cat_results) / len(cat_results)
        cat_time = sum(r['latency_s'] for r in cat_results) / len(cat_results)
        
        status = "✅" if cat_passed == len(cat_results) else "🟡" if cat_passed > 0 else "❌"
        print(f"  {status} {cat:<20} {cat_passed}/{len(cat_results)} pass | {cat_coverage:>4.0%} coverage | {cat_time:>4.1f}s avg")
    
    # Difficulty Breakdown
    print("BY DIFFICULTY")
    
    for diff in ['easy', 'medium', 'hard']:
        if diff in by_difficulty:
            diff_results = by_difficulty[diff]
            diff_passed = sum(1 for r in diff_results if r['success'])
            diff_coverage = sum(r['coverage'] for r in diff_results) / len(diff_results)
            diff_time = sum(r['latency_s'] for r in diff_results) / len(diff_results)
            
            status = "✅" if diff_passed == len(diff_results) else "🟡" if diff_passed > 0 else "❌"
            print(f"  {status} {diff.capitalize():<10} {diff_passed}/{len(diff_results)} pass | {diff_coverage:>4.0%} coverage | {diff_time:>4.1f}s avg")
    
    # Detailed Failures
    failures = [r for r in results if not r['success']]
    if failures:
        print("FAILED QUERIES")
        for i, q in enumerate(EVAL_QUERIES):
            if not results[i]['success']:
                print(f"  • {q['question']}")
                print(f"    Coverage: {results[i]['coverage']:.0%} | Missing: {results[i]['total'] - len(results[i]['found'])} keywords")
    
    # Grade
    print("OVERALL GRADE")
    
    success_rate = passed / len(results)
    
    if success_rate >= 0.9 and avg_coverage >= 0.7 and avg_time < 8:
        grade = "A+ Excellent"
    elif success_rate >= 0.8 and avg_coverage >= 0.65 and avg_time < 10:
        grade = "A - Very Good"
    elif success_rate >= 0.7 and avg_coverage >= 0.6 and avg_time < 12:
        grade = "B - Good"
    elif success_rate >= 0.6 and avg_coverage >= 0.5 and avg_time < 15:
        grade = "C - Acceptable"
    elif success_rate >= 0.5:
        grade = "D - Needs Work"
    else:
        grade = "F - Poor"
    
    print(f"\nGrade: {grade}")
    
    # Strengths & Weaknesses
    print(f"\n  Strengths:")
    if avg_time < 8:
        print(f"    ✓ Fast response times ({avg_time:.1f}s)")
    if avg_coverage >= 0.65:
        print(f"    ✓ Good information coverage ({avg_coverage:.0%})")
    if success_rate >= 0.8:
        print(f"    ✓ High success rate ({success_rate:.0%})")
    
    print(f"\n  Areas for Improvement:")
    if avg_time >= 10:
        print(f"    ✗ Response time could be faster ({avg_time:.1f}s)")
    if avg_coverage < 0.6:
        print(f"    ✗ Coverage needs improvement ({avg_coverage:.0%})")
    if success_rate < 0.8:
        print(f"    ✗ Success rate below target ({success_rate:.0%})")
    
    # Hard query performance
    if 'hard' in by_difficulty:
        hard_passed = sum(1 for r in by_difficulty['hard'] if r['success'])
        hard_total = len(by_difficulty['hard'])
        if hard_passed < hard_total:
            print(f"    ✗ Struggles with complex queries ({hard_passed}/{hard_total})")
    


if __name__ == "__main__":
    main()