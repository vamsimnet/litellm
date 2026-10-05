"""Set FASTROUTER_API_KEY, then run this file for a streaming chat completion."""

import sys
from typing import Final

import litellm


def main() -> None:
    response: Final = litellm.completion(
        model="fastrouter/openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "Tell me about the scaling laws of memory"}],
        stream=True,
    )
    for chunk in response:
        sys.stdout.write(chunk.choices[0].delta.content or "")
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
