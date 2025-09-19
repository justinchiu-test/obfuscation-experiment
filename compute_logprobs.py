import os
import json
from typing import List, Dict, Any
from together import Together


def read_file(filepath: str) -> str:
    """Read content from a file."""
    with open(filepath, 'r') as f:
        return f.read()


def get_total_logprob(client: Together, text: str, model: str) -> tuple[float, float]:
    """
    Get total log probability for the entire text using prompt echoing.

    Args:
        client: Together AI client
        text: Input text to analyze
        model: Model to use for generation

    Returns:
        Total log probability
        Number of tokens
    """
    # Use completion API with echo=True to get log probs for the input tokens
    response = client.completions.create(
        model=model,
        prompt=text,
        max_tokens=1,  # We only need the input tokens
        echo=True,     # Return the prompt tokens with their log probs
        logprobs=1,    # Return log probabilities
        temperature=0  # Deterministic
    )
    assert text == "".join(response.prompt[0].logprobs.tokens)
    total_logprob = sum(response.prompt[0].logprobs.token_logprobs[1:])
    return total_logprob, len(response.prompt[0].logprobs.tokens)


def compare_files(client: Together, model: str, file1: str, file2: str, comparison_name: str):
    """Compare two files and print results."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {comparison_name}")
    print(f"{'='*60}")

    results = []
    for filepath in [file1, file2]:
        if os.path.exists(filepath):
            print(f"\nAnalyzing {filepath}...")
            content = read_file(filepath)
            total_logprob, tokens = get_total_logprob(client, content, model)

            result = {
                'file': filepath,
                'total_logprob': total_logprob,
                'tokens': tokens,
            }
            results.append(result)

            print(f"  Total log probability: {total_logprob:.4f}")
            print(f"  Number of tokens: {tokens}")
        else:
            print(f"File not found: {filepath}")
            return None

    if len(results) == 2:
        file1_res, file2_res = results[0], results[1]
        print(f"\n{file1_res['file']}:")
        print(f"  Total log probability: {file1_res['total_logprob']:.4f}")
        print(f"  Number of tokens: {file1_res['tokens']}")

        print(f"\n{file2_res['file']}:")
        print(f"  Total log probability: {file2_res['total_logprob']:.4f}")
        print(f"  Number of tokens: {file2_res['tokens']}")

        # Token reduction
        token_reduction = file1_res['tokens'] - file2_res['tokens']
        token_reduction_pct = (token_reduction / file1_res['tokens']) * 100
        print(f"\n  Token reduction: {token_reduction} ({token_reduction_pct:.1f}%)")

        if file1_res['total_logprob'] > file2_res['total_logprob']:
            print(f"  → {file1_res['file']} has HIGHER total log probability")
        else:
            print(f"  → {file2_res['file']} has HIGHER total log probability")

    return results


def main():
    """Main function to run all 4 comparisons."""

    # Initialize Together client
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        print("Error: Please set TOGETHER_API_KEY environment variable")
        print("You can get an API key from https://api.together.xyz/")
        return

    client = Together(api_key=api_key)

    # Model to use
    model = "deepseek-ai/DeepSeek-V3"

    print(f"Using model: {model}")

    all_results = []

    # Comparison 1: Original modeling_llama vs obfuscated (with docstrings)
    results1 = compare_files(
        client, model,
        'modeling_llama.py',
        'modeling_llama_obfuscated.py',
        'modeling_llama.py vs modeling_llama_obfuscated.py'
    )
    if results1:
        all_results.extend(results1)

    # Comparison 2: No comments modeling_llama vs obfuscated no comments
    results2 = compare_files(
        client, model,
        'modeling_llama_nocomments.py',
        'modeling_llama_obfuscated_nocomments.py',
        'modeling_llama_nocomments.py vs modeling_llama_obfuscated_nocomments.py'
    )
    if results2:
        all_results.extend(results2)

    # Comparison 3: Original transformer_bria vs obfuscated (with docstrings)
    results3 = compare_files(
        client, model,
        'transformer_bria.py',
        'transformer_bria_obfuscated.py',
        'transformer_bria.py vs transformer_bria_obfuscated.py'
    )
    if results3:
        all_results.extend(results3)

    # Comparison 4: No comments transformer_bria vs obfuscated no comments
    results4 = compare_files(
        client, model,
        'transformer_bria_nocomments.py',
        'transformer_bria_obfuscated_nocomments.py',
        'transformer_bria_nocomments.py vs transformer_bria_obfuscated_nocomments.py'
    )
    if results4:
        all_results.extend(results4)

    # Save all results to JSON
    output_file = 'logprob_results_all_comparisons.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()