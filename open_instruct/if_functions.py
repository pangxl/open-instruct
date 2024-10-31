import re
import json
import langdetect
from typing import List

"""
This module contains functions to verify constraints in the responses generated by the model.
It covers all 25 constraints from the IFEval taxonomy. To be used either for eval or for ground truth rewards.
"""



# include keywords: Include keywords {keyword1}, {keyword2} in your response

def verify_keywords(text, keyword_list):
	"""
	Verify if the response contains all the specified keywords.

	Args:
		response (str): The response text to check
		keyword_list (list): A list of keywords to check for

	Returns:
		bool: True if all keywords are present in the response, False otherwise
	"""
	# Convert response to lowercase for case-insensitive matching
	response_lower = text.lower()

	# Check if all keywords are present in the response
	return all(keyword.lower() in response_lower for keyword in keyword_list)


# Keyword Frequency: In your response, the word {word} should appear {N} times.
def verify_keyword_frequency(text, word, N):
	"""
	Verifies if a keyword appears exactly N times in the given text.

	Args:
		text (str): The text to analyze
		keyword (str): The keyword to count
		expected_count (int): The expected number of occurrences

	Returns:
		tuple: (bool, int) - (Whether constraint is met, actual count found)
	"""
	# Convert text to lowercase to make the search case-insensitive
	text = text.lower()
	keyword = word.lower()

	# Split text into words and remove punctuation
	import re
	words = re.findall(r'\b\w+\b', text)

	# Count actual occurrences
	actual_count = sum(1 for word in words if word == keyword)

	# Check if constraint is met
	constraint_met = actual_count == N

	return constraint_met


# Forbidden Words: Do not include keywords {forbidden words} in the response.
def validate_forbidden_words(text, forbidden_words):
	"""
	Validates that the text does not contain any of the specified forbidden words.

	Args:
		text (str): The text to check for forbidden words
		forbidden_words (list[str]): A list of forbidden words

	Returns:
		tuple[bool, list[str]]: A tuple containing:
			- Boolean indicating if any forbidden words are present
			- List of forbidden words found in the text

	Example:
		text = "This is a message that should not contain any bad words"
		forbidden_words = ["bad", "evil", "harmful"]
		result = validate_forbidden_words(text, forbidden_words)
	"""
	# Convert text to lowercase for case-insensitive matching
	text_lower = text.lower()

	# Check each forbidden word
	found_words = [word for word in forbidden_words if word.lower() in text_lower]

	# Return results
	return len(found_words) == 0


# Letter Frequency : In your response, the letter {letter} should appear {N} times.

def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
	"""
	Verifies if a given letter appears exactly the specified number of times in the text.

	Args:
		text (str): The text to check
		letter (str): The letter to count (case-sensitive)
		target_count (int): The expected number of occurrences

	Returns:
		bool: True if the constraint is met, False otherwise

	Example:
		>>> verify_letter_frequency("hello world", "l", 3)
		True
		>>> verify_letter_frequency("hello world", "o", 2)
		True
		>>> verify_letter_frequency("hello world", "x", 0)
		True
	"""
	if len(letter) != 1:
		raise ValueError("Letter parameter must be a single character")

	actual_count = text.count(letter)
	return actual_count == N


# Response Language: Your ENTIRE response should be in {language}, no other language is allowed.

def validate_response_language(text, language):
	"""
	Validates that the entire response is in the specified language.

	Args:
		text (str): The text to check
		language (str): The language code (e.g., 'en' for English)

	Returns:
		bool: True if the response is entirely in the specified language, False otherwise

	Example:
		text = "This is an English sentence"
		language = "en"
		result = validate_response_language(text, language)
	"""
	from langdetect import detect

	# Detect the language of the text
	detected_language = detect(text)
	# Check if the detected language matches the expected language
	return detected_language == language


# Number Paragraphs: Your response should contain {N} paragraphs. You separate paragraphs using the markdown divider:
# * * *
def verify_paragraph_count(text: str, N: int) -> bool:
	"""
	Verifies that a text contains the expected number of paragraphs,
	where paragraphs are separated by markdown dividers '* * *'

	Args:
		text (str): The text to analyze
		expected_count (int): Expected number of paragraphs

	Returns:
		bool: True if the text contains exactly the expected number of paragraphs,
			  False otherwise

	Example:
		 text = "First paragraph\n* * *\nSecond paragraph"
		 verify_paragraph_count(text, 2)
		True
	"""
	def clean_text(text: str) -> str:
		"""Remove extra whitespace and normalize line endings"""
		return '\n'.join(line.strip() for line in text.splitlines()).strip()

	# Clean the input text
	text = clean_text(text)

	# Split text by markdown divider
	# Add 1 to count since n dividers create n+1 paragraphs
	paragraphs = text.split('* * *')
	actual_count = len(paragraphs)

	# Verify each split resulted in non-empty content
	valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
	if len(valid_paragraphs) != actual_count:
		return False

	return actual_count == N


# Number Words: Answer with at least / around / at most {N} words

def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
	"""
	Validates if a text meets specified word count constraints.

	Args:
		text (str): The text to check
		count (int): The target word count
		qualifier (str): The type of constraint ('at least', 'around', 'at most')

	Returns:
		bool: True if the constraint is met, False otherwise

	Raises:
		ValueError: If an invalid qualifier is provided
	"""
	# Remove extra whitespace and split into words
	words = text.strip().split()
	actual_count = len(words)

	# Define tolerance for "around" qualifier (±10% of target count)
	tolerance = max(round(N * 0.1), 1)

	if quantifier == "at least":
		return actual_count >= N
	elif quantifier == "at most":
		return actual_count <= N
	elif quantifier == "around":
		return abs(actual_count - N) <= tolerance
	else:
		return False


# Number Sentences: Answer with at least / around / at most {N} sentences.
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
	"""
	Verifies if a text contains the expected number of sentences.

	Args:
		text (str): The text to analyze
		N (int): The expected number of sentences
		quantifier (str): The quantifier ('at least', 'around', 'at most')

	Returns:
		bool: True if the text contains the expected number of sentences, False otherwise
	"""
	# Split the text into sentences
	sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

	# Count the number of sentences
	actual_count = len(sentences)

	# Check if the actual count matches the expected count based on the quantifier
	if quantifier == 'at least':
		return actual_count >= N
	elif quantifier == 'around':
		return abs(actual_count - N) <= 1
	elif quantifier == 'at most':
		return actual_count <= N
	else:
		return False


# Number Paragraphs + First Word in i-th Paragraph: There should be {N} paragraphs. Paragraphs and only paragraphs
# are separated with each other by two line breaks. The {i}-th paragraph must start with word {first word}.
def validate_paragraphs(text, N, first_word, i):
	"""
	Validates that a text contains the expected number of paragraphs and that the i-th paragraph starts with a specific
	word.

	Args:
		text (str): The text to analyze
		N (int): The expected number of paragraphs
		first_word (str): The expected first word of the i-th paragraph
		i (int): The index of the paragraph to check (1-indexed)

	Returns:
		bool: True if the text meets the paragraph and first word requirements, False otherwise
	"""
	# Split the text into paragraphs
	paragraphs = text.split('\n\n')

	# Check if the number of paragraphs is as expected
	if len(paragraphs) != N:
		return False

	# Check if the i-th paragraph starts with the specified first word
	if paragraphs[i - 1].strip().startswith(first_word):
		return True
	return False


# Postscript: At the end of your response, please explicitly add a postscript starting with {postscript marker}

def verify_postscript(text, postscript_marker):
	"""
	Verifies if a text contains a postscript starting with '{postscript marker}'

	Args:
		text (str): The text to verify

	Returns:
		bool: True if the text contains a valid postscript, False otherwise
	"""
	# Check if the text contains the postscript marker
	if postscript_marker in text:
		# Get the index of the marker
		marker_index = text.find(postscript_marker)
		# Check if the marker appears near the end
		remaining_text = text[marker_index:].strip()
		# Verify it's not just the marker alone
		return len(remaining_text) > len(postscript_marker)
	return False


# Number Placeholder: The response must contain at least {N} placeholders represented by square brackets,
# such as [address].
def validate_placeholders(text: str, N: int) -> tuple[bool, List[str]]:
	"""
	Validates if a text contains at least the specified number of placeholders in square brackets.

	Args:
		text (str): The text to check for placeholders
		min_placeholders (int): Minimum number of placeholders required

	Returns:
		tuple[bool, List[str]]: A tuple containing:
			- Boolean indicating if the text meets the placeholder requirement
			- List of found placeholders

	Example:
		>>> text = "Hello [name], your [item] will be delivered to [address]"
		>>> validate_placeholders(text, 2)
		(True, ['name', 'item', 'address'])
	"""
	# Find all placeholders using regex
	pattern = r'\[(.*?)\]'
	placeholders = re.findall(pattern, text)

	# Check if the number of placeholders meets the requirement
	has_enough = len(placeholders) >= N

	return has_enough, placeholders


# Number Bullets: Your answer must contain exactly {N} bullet points. Use the markdown bullet points such as: * This
# is a point.
def verify_bullet_points(text: str, N: int) -> tuple[bool, str]:
	"""
	Verifies if a text contains exactly N bullet points in markdown format.
	Returns a tuple of (is_valid, message).

	Args:
		text (str): The text to check
		expected_count (int): The expected number of bullet points

	Returns:
		tuple[bool, str]: (True if constraint is met, explanation message)
	"""
	# Split text into lines and count lines starting with * or -
	lines = text.split('\n')
	bullet_points = [line.strip() for line in lines if line.strip().startswith(('*', '-'))]
	actual_count = len(bullet_points)

	if actual_count == N:
		return True
	else:
		return False


# Title: Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.
def validate_title(text: str) -> bool:
	pattern = r"<<(.*?)>>"
	matches = re.findall(pattern, text)

	if len(matches) > 0:
		return True
	else:
		return False


# Choose: From Answer with one of the following options: {options}
def validate_choice(text: str, options: list) -> bool:
	for option in options:
		if text in option:
			return True
	return False


# Minimum Number Highlighted Section: Highlight at least {N} sections in your answer with markdown, i.e. *highlighted
# section*
def validate_highlighted_sections(text: str, N: int) -> bool:
	pattern = r"\*(.*?)\*"
	matches = re.findall(pattern, text)

	if len(matches) >= N:
		return True
	else:
		return False


# Multiple Sections: Your response must have {N} sections. Mark the beginning of each section with {section splitter} X.

def validate_sections(text: str, N: int, section_splitter: str) -> bool:
	sections = text.split(section_splitter)
	# The first section might not start with the splitter, so we adjust for this
	if sections[0] == '':
		sections.pop(0)
	if len(sections) == N:
		return True
	else:
		return False


# JSON Format : Entire output should be wrapped in JSON format.
def validate_json_format(text: str) -> bool:
	try:
		json_object = json.loads(text)
	except ValueError as e:
		return False
	return True


# Repeat Prompt: First, repeat the request without change, then give your answer (do not say anything before
# repeating the request; the request you need to repeat does not include this sentence)
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
	if text.startswith(original_prompt):
		return True
	else:
		return False


# Two Responses: Give two different responses. Responses and only responses should be separated by 6 asterisk
# symbols: ******.
def validate_two_responses(text: str) -> bool:
	if text.count('******') == 1:
		response_list = text.split('******')
		first_response = response_list[0].strip()
		second_response = response_list[1].strip()
		if first_response != second_response:
			return True
	return False


# All Uppercase: Your entire response should be in English, capital letters only.
def validate_uppercase(text: str) -> bool:
	# Check if the response is the same as the uppercase version of the response
	if text == text.upper():
		return True
	else:
		return False


# All Lowercase: Your entire response should be in English, and in all lowercase letters. No capital letters are
# allowed.
def validate_lowercase(text: str) -> bool:
	# Check if the response is the same as the lowercase version of the response
	if text == text.lower():
		return True
	else:
		return False


# Frequency of All-capital Words: In your response, words with all capital letters should appear at least / around /
# at most {N} times.
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
	words = re.findall(r'\b[A-Z]+\b', text)
	if quantifier == 'at least':
		return len(words) >= N
	elif quantifier == 'around':
		return len(words) == N
	elif quantifier == 'at most':
		return len(words) <= N
	else:
		return False


# End Checker: Finish your response with this exact phrase {end phrase}. No other words should follow this phrase.
def validate_end(text: str, end_phrase: str) -> bool:
	# Check if the response ends with the end phrase
	if text.endswith(end_phrase):
		return True
	else:
		return False


# Quotation: Wrap your entire response with double quotation marks.
def validate_quotation(text: str) -> bool:
	if text.startswith('"') and text.endswith('"'):
		return True
	else:
		return False


# No Commas: In your entire response, refrain from the use of any commas.
def validate_no_commas(text: str) -> bool:
	if ',' not in text:
		return True
	else:
		return False

IF_FUNCTIONS_MAP = {
	'verify_keywords': verify_keywords,
	'verify_keyword_frequency': verify_keyword_frequency,
	'validate_forbidden_words': validate_forbidden_words,
	'verify_letter_frequency': verify_letter_frequency,
	'validate_response_language': validate_response_language,
	'verify_paragraph_count': verify_paragraph_count,
	'validate_word_constraint': validate_word_constraint,
	'verify_sentence_constraint': verify_sentence_constraint,
	'validate_paragraphs': validate_paragraphs,
	'verify_postscript': verify_postscript,
	'validate_placeholders': validate_placeholders,
	'verify_bullet_points': verify_bullet_points,
	'validate_title': validate_title,
	'validate_choice': validate_choice,
	'validate_highlighted_sections': validate_highlighted_sections,
	'validate_sections': validate_sections,
	'validate_json_format': validate_json_format,
	'validate_repeat_prompt': validate_repeat_prompt,
	'validate_two_responses': validate_two_responses,
	'validate_uppercase': validate_uppercase,
	'validate_lowercase': validate_lowercase,
	'validate_frequency_capital_words': validate_frequency_capital_words,
	'validate_end': validate_end,
	'validate_quotation': validate_quotation,
	'validate_no_commas': validate_no_commas
}
