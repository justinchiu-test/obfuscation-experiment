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


def main():
    """Main function to analyze simple.py and simple_small.py"""

    # Initialize Together client
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        print("Error: Please set TOGETHER_API_KEY environment variable")
        print("You can get an API key from https://api.together.xyz/")
        return

    client = Together(api_key=api_key)

    # Files to analyze
    #files = ['simple.py', 'simple_small.py']
    files = ['modeling_llama_nocomments.py', 'modeling_llama_obfuscated_nocomments.py']

    # Model to use
    model = "deepseek-ai/DeepSeek-V3"

    print(f"Using model: {model}")

    results = []
    for filepath in files:
        if os.path.exists(filepath):
            print(f"\nAnalyzing {filepath}...")

            # Read file content
            content = read_file(filepath)

            # Get total log probability
            total_logprob, tokens = get_total_logprob(client, content, model)

            result = {
                'file': filepath,
                'total_logprob': total_logprob,
                "tokens": tokens,
            }
            results.append(result)

            print(f"  Total log probability: {total_logprob:.4f}")
            print(f"  Number of tokens: {tokens}")
        else:
            print(f"File not found: {filepath}")

    # Save results to JSON
    output_file = 'logprob_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Compare the two files
    if len(results) == 2:
        print("\n" + "="*50)
        print("COMPARISON")
        print("="*50)

        file1, file2 = results[0], results[1]
        print(f"\n{file1['file']}:")
        print(f"  Total log probability: {file1['total_logprob']:.4f}")
        print(f"  Number of tokens: {file1['tokens']}")

        print(f"\n{file2['file']}:")
        print(f"  Total log probability: {file2['total_logprob']:.4f}")
        print(f"  Number of tokens: {file2['tokens']}")

        if file1['total_logprob'] > file2['total_logprob']:
            print(f"\n→ {file1['file']} has HIGHER total log probability")
        else:
            print(f"\n→ {file2['file']} has HIGHER total log probability")

if __name__ == "__main__":
    main()
