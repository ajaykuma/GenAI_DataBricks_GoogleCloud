"""
Short Story Generator using Anthropic Claude API
=================================================
This script takes user input and generates short stories
with configurable parameters to control output generation.

Requirements:
    pip install anthropic

Setup:
    Set your API key as an environment variable:
    export ANTHROPIC_API_KEY="your-api-key-here"
"""

import anthropic
import os

# ─────────────────────────────────────────────
# PARAMETERS — tweak these to control output
# ─────────────────────────────────────────────

PARAMETERS = {
    "model": "claude-haiku-4-5-20251001",   # Model to use (haiku = fast & cheap)
    "max_tokens": 500,                        # Maximum length of the story (higher = longer)
    "temperature": 1.0,                       # Creativity level: 0.0 (focused) → 1.0 (creative)
    "top_p": 0.9,                             # Token diversity: lower = safer, higher = varied
    "top_k": 50,                              # Limit to top-k tokens at each step
    "story_length": "short",                  # Hint to model: "short", "medium", "long"
    "tone": "fun and adventurous",            # Story tone/mood
    "genre": "fantasy",                       # Genre: fantasy, sci-fi, mystery, romance, etc.
    "target_audience": "children",            # children, teens, adults
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT — defines the model's behavior
# ─────────────────────────────────────────────

def build_system_prompt(params: dict) -> str:
    return f"""You are a creative story writer. Your job is to write engaging {params['story_length']} stories.

Follow these rules:
- Genre: {params['genre']}
- Tone: {params['tone']}
- Target audience: {params['target_audience']}
- Keep the story concise but complete with a clear beginning, middle, and end.
- Make it imaginative and captivating.
- Do not add titles or labels — just tell the story directly.
"""

# ─────────────────────────────────────────────
# STORY GENERATION FUNCTION
# ─────────────────────────────────────────────

def generate_story(user_prompt: str, params: dict) -> str:
    """
    Generate a short story based on user input and defined parameters.

    Args:
        user_prompt: The story idea or theme from the user
        params: Dictionary of generation parameters

    Returns:
        Generated story as a string
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model=params["model"],
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        system=build_system_prompt(params),
        messages=[
            {
                "role": "user",
                "content": f"Write a {params['story_length']} {params['genre']} story about: {user_prompt}"
            }
        ]
    )

    return message.content[0].text

# ─────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("      SHORT STORY GENERATOR (Claude AI)")
    print("=" * 50)
    print(f"  Genre        : {PARAMETERS['genre']}")
    print(f"  Tone         : {PARAMETERS['tone']}")
    print(f"  Audience     : {PARAMETERS['target_audience']}")
    print(f"  Creativity   : {PARAMETERS['temperature']} / 1.0")
    print(f"  Max tokens   : {PARAMETERS['max_tokens']}")
    print("=" * 50)

    while True:
        print("\nEnter your story idea (or type 'quit' to exit):")
        user_input = input(">> ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! Happy storytelling! 👋")
            break

        if not user_input:
            print("Please enter a story idea.")
            continue

        print("\n Generating your story...\n")

        try:
            story = generate_story(user_input, PARAMETERS)
            print("─" * 50)
            print(story)
            print("─" * 50)
        except anthropic.AuthenticationError:
            print(" Error: Invalid API key. Please check your ANTHROPIC_API_KEY.")
            break
        except anthropic.RateLimitError:
            print(" Error: Rate limit reached. Please wait and try again.")
        except Exception as e:
            print(f" Unexpected error: {e}")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
