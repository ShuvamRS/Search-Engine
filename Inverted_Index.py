# Author: Shuvam Raj Satyal

import sys
import os
import json
import pickle
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import snowball


class Posting:
	def __init__(self, docid, tf, fields, termPosition=None):
		self.docid = docid
		self.tf = tf # term frequency
		self.fields = fields
		self.termPositions = []
		if termPosition: self.append_term_position(termPosition)

	def append_term_position(self, termPosition):
		self.termPositions.append(termPosition)


class PostingList:
	def __init__(self, posting=None):
		self.posting_list = []
		self.df = 0 # document frequency
		if posting: self.append(posting)

	def __getitem__(self, docid):
		# implement indexing by docid
		for i, posting in enumerate(self.posting_list):
			if docid == posting.docid:
				return posting
		raise IndexError

	def append(self, posting):
		self.posting_list.append(posting)
		self.df += 1
		# sort posting_list by docid
		self.posting_list.sort(key=lambda x: x.docid)



def get_document_paths(corpus_path):
	# Returns document paths inside specified corpus.

	document_paths = []
	# Get list of all directories in corpus
	directories = os.listdir(corpus_path)

	for directory in directories:
		directory_path = os.path.join(corpus_path, directory)
		try:
			paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
			document_paths.extend(paths)

		except NotADirectoryError:
			# Ignore .DS_Store
			continue

	return tuple(set(document_paths))



def tokenize(text):
	# Tokenize text based on tokenize_pattern.
	# Returns list contains duplicates.

	tokenize_pattern = r"[a-zA-Z0-9]+"
	tokenizer = RegexpTokenizer(tokenize_pattern)
	return tokenizer.tokenize(text)



def get_token_frequency(tokens):
	freq = {}
	for token in tokens:
		try:
			freq[token] += 1
		except KeyError:
			freq[token] = 1
	return freq



def get_HTML_tag_fields(soup):
	# Tokenizes text inside HTML tags like title, i, b, em, strong
	# Returns a dictionary with key=tag names and value=set of tokens

	HTML_tag_fields = {"title": set(), "heading": set(), "bold": set(), "strong": set(), "italics":set(), "emphasized": set()}
	tag_fields = {"h1":"heading", "h2":"heading", "h3":"heading", "b":"bold", "strong":"strong", "i":"italics", "em":"emphasized"}

	# Get title tokens if title exists
	if soup.title != None and isinstance(soup.title.string, str):
		HTML_tag_fields["title"] = set(tokenize(soup.title.string))

	tags = tag_fields.keys()
	for elem in soup.find_all(tags):
		tag_set = set(tokenize(elem.get_text()))
		HTML_tag_fields[tag_fields[elem.name]] = HTML_tag_fields[tag_fields[elem.name]].union(tag_set)

	return HTML_tag_fields



def get_posting_fields(HTML_tag_fields, token):
	# Checks if token is present inside HTML tags like title, i, b, em, strong
	# based on the dictionary created by get_HTML_tag_fields().

	posting_fields = {"title": False, "heading": False, "bold": False, "strong":False, "italics": False, "emphasized":False}
	
	for field, words in HTML_tag_fields.items():
		if token in words:
			posting_fields[field] = True

	return posting_fields
	
	

def BuildInvertedIndex(document_paths):
	# In-memory indexer for creating inverted index
	
	stemmer = snowball.SnowballStemmer('english')
	DocumentIndex = {} # {key = doc_id: value = (url, doc_path)}
	InvertedIndex = {} # Inverted list storage (dictionary of tokens/words/n-grams + posting lists)
	n = 0 # Document numbering

	for document_path in document_paths:
		# Read json file which contains ['url', 'content', 'encoding'] for a document
		with open(document_path, 'r') as fh:
			json_object = json.load(fh)
		
		url, pageContent, encoding = json_object["url"], json_object["content"], json_object["encoding"]
		
		# Ignore urls with fragments
		if urlparse(url).fragment != "": continue

		soup = BeautifulSoup(pageContent, 'lxml')
		text = soup.get_text()

		# check if the page contains any text
		if text == '': continue
		n += 1
		print(f"Indexing document #{n}")

		DocumentIndex[n] = (url, document_path)
		tokens = tokenize(text) # tokenize text in html document
		tokenFrequency = get_token_frequency(tokens)
		tokens = set(tokens) # remove duplicate tokens

		HTML_tag_fields = get_HTML_tag_fields(soup)

		for term_position, token in enumerate(tokens):
			# Check if a PostingList is present in the inverted index,
			# Add the new {token : PostingList} to inverted index otherwise.
			try:
				posting_list = InvertedIndex[stemmer.stem(token)]
				# Check if a Posting is present in the posting_list,
				# Add the new Posting to posting_list otherwise.
				try:
					# If Posting 'n' is present in the posting_list,
					# Append term position to posting.term_postitions
					posting = posting_list[n]
					posting.append_term_position(term_position)

				except IndexError:
					# if Posting 'n' is not present in the posting_list
					posting_list.append(Posting(docid=n, tf=tokenFrequency[token], fields=get_posting_fields(HTML_tag_fields, token), termPosition=term_position))

			except KeyError:
				InvertedIndex[stemmer.stem(token)] = PostingList(Posting(docid=n, tf=tokenFrequency[token], fields=get_posting_fields(HTML_tag_fields, token), termPosition=term_position))

	return DocumentIndex, InvertedIndex



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print(f"Expected 2 arguments(File Paths) received {len(sys.argv)-1} argument(s) instead")
		raise SystemExit

	corpus_path = sys.argv[1]
	storage_path = sys.argv[2]


	document_paths = get_document_paths(corpus_path)
	DocumentIndex, InvertedIndex = BuildInvertedIndex(document_paths)

	# Store DocumentIndex on disk for future retrieval
	with open(os.path.join(storage_path, "DocIndex.json"), 'w') as fh:
		json.dump(DocumentIndex, fh, indent=3)

	# Store InvertedIndex on disk
	with open(os.path.join(storage_path, "InvIndex.pickle"), 'wb') as fh:
		pickle.dump(InvertedIndex, fh)