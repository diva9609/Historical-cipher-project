import jiwer


def read_file(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return content


def compute_cer(ground_truth_path, prediction_path):

    reference_str = read_file(ground_truth_path)
    hypothesis_str = read_file(prediction_path)


    reference_count = len(reference_str.split())

    output = jiwer.process_words(reference=reference_str, hypothesis=hypothesis_str)

    error_rate = output.wer

    print(f"Ground Truth: {ground_truth_path}")
    print(f"Prediction:   {prediction_path}")
    print(f"Total reference symbols: {reference_count}")
    print(f"Symbol Error Rate (SER): {error_rate:.4f} ({error_rate*100:.2f}%)")
    print("-" * 40)
    
    return error_rate


if __name__ == "__main__":
    compute_cer("gold/Ramanacoil/test_transcript.txt",
                "4LabelPropagation/Ramanacoil_x_15/label_propagation_final_string.txt")