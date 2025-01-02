from backend.core.db.db_layer import get_test_results, get_test_results_detail, get_test_names


def getTestResults(testId, mode, test_result_id):
    
    df_detail = get_test_results(testId)
    # Get the first record as a dictionary
    summary = df_detail.to_dict('records')[0]
    print(f"summary-{summary}")

    # Get results as a DataFrame
    df = get_test_results_detail(testId,test_result_id)
    
    # Transform DataFrame into required format
    transformed_results = {
        f"test{idx}": {
            "ideal_response": row["original_response"] if summary.get("test_type") == "consistency" else row["ideal_response"],
            "actual_response": row["actual_response"],
            "original_prompt": row["original_prompt"],
            "test_results_detail_no": row["test_results_detail_no"],
            "trd_fingerprint": row["trd_fingerprint"],
            "rs_fingerprint": row["rs_fingerprint"],
            "matched_tokens": row["matched_tokens"],
            "mismatched_tokens": row["mismatched_tokens"],
            "total_tokens": row["matched_tokens"] + row["mismatched_tokens"],
            "mismatch_percentage": round(float(row["mismatch_percentage"]),2),
            "execution_time": row["execution_time"],
            "page": row["page"],
            "status": row["status"],
            "test_run_no": row["test_run_no"]
        }
        for idx, row in df.iterrows()
    }
    
    results = {
        "transformed_results": transformed_results,
        "summary": summary
    }
    print(f"results-{results}")
    return results

def get_eval_names():
    df = get_test_names()
    
    return df.to_dict('records')

def get_test_result_details(test_run_no):
    df = get_test_results_detail(test_run_no)
    return df.to_dict('records')

#if __name__ == "__main__":
#    print(getTestResults(17))
