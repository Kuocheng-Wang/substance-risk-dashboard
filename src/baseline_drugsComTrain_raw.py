import pandas as pd
import re
from pathlib import Path


def normalize_text(text):
    return str(text).lower().strip()


def contains_any_phrase(text, phrases):
    text = normalize_text(text)
    return any(p in text for p in phrases)


def contains_any_word(text, words):
    """
    Match with word boundaries to reduce false positives.
    """
    text = normalize_text(text)
    for w in words:
        pattern = r"\b" + re.escape(w) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def contains_any_pattern(text, patterns):
    text = normalize_text(text)
    return any(re.search(p, text) for p in patterns)


# -----------------------------
# 1. General medical-context exclusions
# -----------------------------
MEDICAL_EXCLUDE_PHRASES = [
    "birth control",
    "allergy",
    "allergies",
    "bp meds",
    "blood pressure meds",
    "blood pressure medication",
    "colonoscopy prep",
    "prep for colonoscopy",
    "colon prep",
    "symptoms returned",
    "returned symptoms",
    "side effects almost gone",
    "feel normal again",
    "life saver",
    "changed my life",
    "no longer have side effects",
    "almost gone",
    "pain medication after surgery",
    "post surgery medication",
    "surgery recovery",
    "antibiotic",
    "antibiotics",
    "insulin",
    "vitamin",
    "vitamins",
    "thyroid medication",
    "asthma medication",
    "allergy medicine",
    "allergy medication",
]

MEDICAL_EXCLUDE_PATTERNS = [
    r"\bfell back onto the bed\b",
    r"\bfell back on the bed\b",
    r"\bsymptoms? returned\b",
    r"\breturned to normal\b",
    r"\brecovered from surgery\b",
    r"\brecovering from surgery\b",
]


# -----------------------------
# 2. Substance context terms
# Generic terms like pill/medication/drug are intentionally excluded
# -----------------------------
SUBSTANCE_WORDS = [
    "opioid", "opioids",
    "oxy", "oxycodone",
    "heroin",
    "cocaine",
    "weed",
    "marijuana",
    "alcohol",
    "suboxone",
    "methadone",
    "fentanyl",
]

ABUSE_CONTEXT_WORDS = [
    "addiction",
    "addicted",
    "withdrawal",
    "withdrawals",
    "detox",
    "craving",
    "cravings",
    "relapse",
    "relapsed",
    "sober",
    "clean",
    "hooked",
    "abuse",
    "misuse",
    "rehab",
    "recovery",
]

ABUSE_CONTEXT_PHRASES = [
    "can't stop",
    "cannot stop",
    "using again",
    "used again",
    "drinking again",
    "drank again",
    "went back to using",
    "started using again",
    "started drinking again",
    "couldn't stay sober",
    "cannot stay sober",
    "dependent on",
]


# -----------------------------
# 3. Relapse context terms
# again/back/return/recover should not trigger by themselves
# -----------------------------
RELAPSE_TRIGGER_WORDS = [
    "relapse",
    "relapsed",
    "withdrawal",
    "withdrawals",
    "recovery",
    "rehab",
    "detox",
]

RELAPSE_TRIGGER_PHRASES = [
    "using again",
    "used again",
    "drinking again",
    "drank again",
    "went back to using",
    "started using again",
    "started drinking again",
    "broke sobriety",
    "couldn't stay sober",
    "cannot stay sober",
    "not sober anymore",
]

RELAPSE_SCENE_WORDS = [
    "opioid", "opioids",
    "oxy", "oxycodone",
    "heroin",
    "cocaine",
    "weed",
    "marijuana",
    "alcohol",
    "suboxone",
    "methadone",
    "fentanyl",
    "addiction",
    "addicted",
    "sober",
    "clean",
    "craving",
    "cravings",
    "abuse",
    "misuse",
]

RELAPSE_SCENE_PHRASES = [
    "can't stop",
    "cannot stop",
    "using again",
    "used again",
    "drinking again",
    "drank again",
    "went back to using",
    "started using again",
    "started drinking again",
]


# -----------------------------
# 4. Distress terms + exclusion terms
# -----------------------------
DISTRESS_WORDS = [
    "depressed",
    "depression",
    "anxiety",
    "anxious",
    "stressed",
    "stress",
    "hopeless",
    "overwhelmed",
    "panic",
    "sad",
    "crying",
    "suicidal",
    "miserable",
    "lonely",
    "afraid",
    "empty",
    "worthless",
]

DISTRESS_EXCLUDE_PHRASES = [
    "no longer",
    "almost gone",
    "feel normal again",
    "life saver",
    "changed my life",
    "no longer anxious",
    "no longer depressed",
    "anxiety is gone",
    "depression is gone",
    "i feel normal again",
]

DISTRESS_EXCLUDE_PATTERNS = [
    r"\bno longer\b",
    r"\balmost gone\b",
    r"\bfeel normal again\b",
    r"\blife saver\b",
    r"\bchanged my life\b",
]


def is_medical_excluded(text):
    text = normalize_text(text)
    if contains_any_phrase(text, MEDICAL_EXCLUDE_PHRASES):
        return True
    if contains_any_pattern(text, MEDICAL_EXCLUDE_PATTERNS):
        return True
    return False


def substance_rule(text):
    """
    substance_label rule:
    1. Exclude general medical-review contexts
    2. Require both:
       - substance-related terms
       - abuse/addiction-related context
    """
    text = normalize_text(text)

    if is_medical_excluded(text):
        return 0

    has_substance = contains_any_word(text, SUBSTANCE_WORDS)
    has_abuse_context = (
        contains_any_word(text, ABUSE_CONTEXT_WORDS)
        or contains_any_phrase(text, ABUSE_CONTEXT_PHRASES)
    )

    return int(has_substance and has_abuse_context)


def relapse_rule(text):
    """
    relapse_label rule:
    1. Exclude general medical-review contexts
    2. Require both:
       - substance/addiction scene terms
       - relapse/withdrawal/recovery trigger terms
    3. again/back/return/recover cannot trigger by themselves
    """
    text = normalize_text(text)

    if is_medical_excluded(text):
        return 0

    has_relapse_scene = (
        contains_any_word(text, RELAPSE_SCENE_WORDS)
        or contains_any_phrase(text, RELAPSE_SCENE_PHRASES)
    )

    has_relapse_trigger = (
        contains_any_word(text, RELAPSE_TRIGGER_WORDS)
        or contains_any_phrase(text, RELAPSE_TRIGGER_PHRASES)
    )

    return int(has_relapse_scene and has_relapse_trigger)


def distress_rule(text):
    """
    distress_label rule:
    1. Match distress-related terms
    2. Exclude clear recovery/improvement expressions
    """
    text = normalize_text(text)

    has_distress = contains_any_word(text, DISTRESS_WORDS)

    has_exclude = (
        contains_any_phrase(text, DISTRESS_EXCLUDE_PHRASES)
        or contains_any_pattern(text, DISTRESS_EXCLUDE_PATTERNS)
    )

    return int(has_distress and not has_exclude)


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "outputs" / "cleanned"/"cleaned_drugsComTrain.csv"
    output_dir = project_root / "outputs" / "tables"
    output_file = output_dir /"predictions_drugsComTrain.csv"
    print("Reading file:", input_file)

    df = pd.read_csv(input_file)

    if "text" not in df.columns:
        raise ValueError("The input file does not contain a 'text' column. Please check cleaned_data.csv.")

    # Generate labels
    df["substance_label"] = df["text"].apply(substance_rule)
    df["distress_label"] = df["text"].apply(distress_rule)
    df["relapse_label"] = df["text"].apply(relapse_rule)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print("predictions_baseline.csv has been generated successfully!")
    print(df.head())

    print("\nLabel statistics:")
    print("Number of rows with substance_label = 1:", int(df["substance_label"].sum()))
    print("Number of rows with distress_label = 1:", int(df["distress_label"].sum()))
    print("Number of rows with relapse_label = 1:", int(df["relapse_label"].sum()))

    print("\nOutput file:", output_file)


if __name__ == "__main__":
    main()