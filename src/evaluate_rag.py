from rag_pipeline import rag_pipeline

# List of evaluation questions
evaluation_questions = [
    "What are the most common complaints about credit card fees?",
    "How do customers describe issues with personal loan approvals?",
    "What problems do users report with savings account interest rates?",
    "What are typical complaints regarding money transfer delays?",
    "How do complaints about credit card billing errors usually go?",
    "What issues arise with personal loan customer service?",
    "What do customers say about savings account withdrawal limits?",
    "How are money transfer fees perceived by customers?",
    "What are the main concerns with credit card security?",
    "How do users complain about personal loan terms and conditions?"
]

def run_evaluation():
    results = []
    for question in evaluation_questions:
        try:
            answer, sources = rag_pipeline(question)
            # For evaluation, manually assess later, but here placeholder
            quality_score = "N/A"  # To be filled manually
            comments = "Automated generation; manual review needed"
            results.append({
                "Question": question,
                "Generated Answer": answer,
                "Retrieved Sources": sources[:2],  # Show first 2
                "Quality Score": quality_score,
                "Comments": comments
            })
        except Exception as e:
            results.append({
                "Question": question,
                "Generated Answer": f"Error: {str(e)}",
                "Retrieved Sources": [],
                "Quality Score": "N/A",
                "Comments": "Pipeline failed"
            })

    # Print markdown table
    print("| Question | Generated Answer | Retrieved Sources | Quality Score | Comments |")
    print("|----------|------------------|-------------------|---------------|----------|")
    for res in results:
        sources_str = "; ".join(res["Retrieved Sources"]) if res["Retrieved Sources"] else "None"
        print(f"| {res['Question']} | {res['Generated Answer'][:100]}... | {sources_str[:200]}... | {res['Quality Score']} | {res['Comments']} |")

if __name__ == "__main__":
    run_evaluation()