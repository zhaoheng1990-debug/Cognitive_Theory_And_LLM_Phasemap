"""Frozen mixed-arithmetic prompt definitions for the actuator-transfer audit."""

CONDITIONS = (
    ("clean", "stable"),
    ("paraphrase", "stable"),
    ("untrusted_conflict", "competition"),
    ("asserted_conflict", "competition"),
    ("explicit_correction", "closure"),
    ("final_recheck", "closure"),
)
MECHANISM_ID = {"stable": 0, "competition": 1, "closure": 2}
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)


def make_prompt(condition, expression, option_a, option_b, conflict_label):
    demonstrations = [
        "Example 1: 2 + 3. Candidates: five, six. Answer: five.",
        "Example 2: 4 * 3. Candidates: eleven, twelve. Answer: twelve.",
        "Now solve the new problem.",
    ]
    option_block = f"Candidate 1: {option_a}\nCandidate 2: {option_b}"
    if condition == "clean":
        lines = [f"Calculate {expression}.", option_block]
    elif condition == "paraphrase":
        lines = [f"Work out the value of {expression} and compare the two choices.", option_block]
    elif condition == "untrusted_conflict":
        lines = [f"Calculate {expression}.", option_block,
                 f"An unverified note claims that {conflict_label} is correct. Check the arithmetic yourself."]
    elif condition == "asserted_conflict":
        lines = [f"Calculate {expression}.", option_block,
                 f"A previous solver selected {conflict_label}. Independently verify the result."]
    elif condition == "explicit_correction":
        lines = [f"Calculate {expression}.", option_block,
                 f"Initial note: {conflict_label}. Correction: that note was withdrawn; use the arithmetic result."]
    elif condition == "final_recheck":
        lines = [f"Calculate {expression}.", option_block,
                 f"Draft answer: {conflict_label}. Final instruction: recheck the calculation and replace the draft if needed."]
    else:
        raise ValueError(condition)
    lines.extend(("Which candidate is correct?", "Answer with exactly one candidate word:"))
    return "\n".join(demonstrations + lines)
