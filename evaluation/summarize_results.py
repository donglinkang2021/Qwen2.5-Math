import json
from pathlib import Path
from tabulate import tabulate

def summarize_evaluation_outputs():
    """
    Parses evaluation output directories to summarize model scores using pathlib,
    accounting for a deeper directory structure, and prints a summary table.
    """
    # base_output_dir = Path(".cache/outputs1")
    base_output_dir = Path("outputs")
    # base_output_dir = Path("outputs/data2/linkdom/converted_model_safetensors")
    if not base_output_dir.is_dir():
        print(f"Error: Base output directory '{base_output_dir}' not found.")
        print(f"Please ensure this script is run from the 'evaluation' directory,")
        print(f"and the '{base_output_dir}' directory exists within it.")
        return

    org_dirs = [d for d in base_output_dir.iterdir() if d.is_dir()]

    if not org_dirs:
        print(f"No organization directories found in '{base_output_dir}'.")
        return

    all_models_summary_data = []
    all_dataset_names = set()

    for org_path in sorted(org_dirs):
        org_name = org_path.name
        
        model_name_dirs = [d for d in org_path.iterdir() if d.is_dir()]
        if not model_name_dirs:
            # print(f"No model directories found in '{org_path}'.") # Optional: keep for debugging
            continue

        for model_name_path in sorted(model_name_dirs):
            model_name = model_name_path.name
            # Temporarily print to show progress, will be replaced by table
            # print(f"Processing: Organization: {org_name}, Model: {model_name}")

            math_eval_path = model_name_path / "math_eval"
            if not math_eval_path.is_dir():
                # print(f"  Info: 'math_eval' directory not found in '{model_name_path}'. Skipping.") # Optional
                continue
            
            current_model_scores = {"org_model_name": f"{org_name}/{model_name}"}
            dataset_scores_for_model = {}
            total_acc_sum = 0.0
            num_datasets_with_scores = 0

            dataset_dirs = [d for d in math_eval_path.iterdir() if d.is_dir()]

            for dataset_path in sorted(dataset_dirs):
                dataset_name = dataset_path.name
                all_dataset_names.add(dataset_name) # Collect all unique dataset names
                
                metrics_file_found = False
                for item in dataset_path.iterdir():
                    if item.is_file() and item.name.endswith("_metrics.json"):
                        metrics_file_path = item
                        try:
                            with metrics_file_path.open('r') as f:
                                data = json.load(f)
                            
                            if "acc" in data:
                                acc_score = data["acc"]
                                dataset_scores_for_model[dataset_name] = acc_score
                                total_acc_sum += acc_score
                                num_datasets_with_scores += 1
                                metrics_file_found = True
                                break 
                        except json.JSONDecodeError:
                            print(f"  Warning: Could not decode JSON from '{metrics_file_path}' for {org_name}/{model_name}")
                        except Exception as e:
                            print(f"  Warning: Error reading '{metrics_file_path}' for {org_name}/{model_name}: {e}")
                
                if not metrics_file_found:
                    # print(f"  Info: No metrics file found for dataset '{dataset_name}' in '{model_name_path / 'math_eval'}'.") # Optional
                    pass


            current_model_scores.update(dataset_scores_for_model)
            if num_datasets_with_scores > 0:
                average_score = total_acc_sum / num_datasets_with_scores
                current_model_scores["Average"] = average_score
            else:
                current_model_scores["Average"] = None # Or "N/A"
            
            all_models_summary_data.append(current_model_scores)

    if not all_models_summary_data:
        print("No data found to tabulate.")
        return

    # Prepare data for tabulate
    sorted_dataset_names = sorted(list(all_dataset_names))
    headers = ["Organization/Model"] + sorted_dataset_names + ["Average"]
    
    table_data = []
    for model_data in all_models_summary_data:
        row = [model_data["org_model_name"]]
        for ds_name in sorted_dataset_names:
            score = model_data.get(ds_name)
            row.append(f"{score:.1f}" if score is not None else "N/A")
        
        avg_score = model_data.get("Average")
        row.append(f"{avg_score:.1f}" if avg_score is not None else "N/A")
        table_data.append(row)

    # sort the table data by average score
    table_data.sort(key=lambda x: (x[-1] if x[-1] != "N/A" else float('-inf')), reverse=True)

    print("\n" + "="*50 + " Summary Table " + "="*50)
    print(tabulate(table_data, headers=headers, tablefmt="github", floatfmt=".1f"))

if __name__ == "__main__":
    summarize_evaluation_outputs()