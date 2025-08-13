Context-Aware Autocorrect Tool
This tool automatically corrects spelling mistakes in text by using context instead of just matching words from a dictionary.
It uses BERT-based language models to understand the meaning of a sentence before suggesting corrections.

Features
Corrects both typos and real-word mistakes (e.g., I like your short → I like your shirt).

Works on single sentences or entire text files.

Uses BERT masked language modeling for context understanding.

Configurable thresholds to control how aggressive the corrections are.

Requirements
Make sure you have Python 3.8+ installed.
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
1. Correct a single sentence
bash
Copy code
python -m src.context_autocorrect --text "I like your short"
Example output:

java
Copy code
I like your shirt
(Edits applied: short → shirt)
2. Correct a file
bash
Copy code
python -m src.context_autocorrect --infile data/mini/clean.txt --outfile corrected.txt
--infile: Path to the input file.

--outfile: Path where corrected text will be saved.

How It Works
Tokenization → Breaks sentences into words and punctuation.

Masking → Replaces a suspected wrong word with [MASK].

Prediction → Uses BERT to predict the most likely replacement word.

Filtering → Applies rules like edit distance, word frequency, and probability thresholds.

Correction → Updates the sentence if the replacement passes all checks.

Example
Input:

css
Copy code
I am goin to the shcool tomorrow.
Output:

css
Copy code
I am going to the school tomorrow.
Project Structure
bash
Copy code
context_aware_autocorrect_tool/
│
├── src/
│   ├── context_autocorrect.py   # Main correction logic
│   ├── mlm_helper.py            # BERT helper functions
│   ├── utils.py                 # Tokenization and text helpers
│
├── data/
│   ├── mini/                    # Sample data files
│
├── requirements.txt
└── README.md
Notes
First run may take time as it downloads the BERT model.

Works better with English text.

You can adjust thresholds in AcceptanceConfig for stricter or looser corrections.

