# Author: Shuvam Raj Satyal

import math
import re
import time
from nltk.corpus import stopwords 
from collections import defaultdict
from Inverted_Index import *
from Search_Cache import Search_Cache

# Patterns to re-construct posting lists from inverted index
INVERTED_INDEX_LINE_PATTERN = r"Term:(?P<term>\w+),PostingList:\[(?P<PostingList>.+)\]\n"
INVERTED_INDEX_POSTINGLIST_PATTERN = r"df:(?P<df>\d+),Postings:\[(?P<postings>.+)\]"
INVERTED_INDEX_POSTING_PATTERN = r"docid:(\d+),tf:(\d+),fields:\[title:(\w+),heading:(\w+),bold:(\w+),strong:(\w+),italics:(\w+),emphasized:(\w+)\],termPositions:\[([0-9,]+)\]"

# Maximum number of postings yielded per query term
RESULT_BATCH_SIZE = 100

# Maximum number of query terms to process
QUERY_THRESHOLD = 10

DISPLAY_URLS_ONLY = True

CACHE = Search_Cache()
STOP_WORDS = set(stopwords.words('english')) 


def get_tf_idf_weight(N, tf, df):
	# N -> # of documents or terms in query, tf -> term frequency, df -> document frequency
	return (1 + math.log10(tf)) * math.log10(N/df)


def compute_cosine_similarity(booleanSearchData, query_words, N):
	cosine_similarity = {}

	# Calculate tf-idf weights of each term in each document
	tf_idf_weight_document = defaultdict(lambda:defaultdict(float))
	dict_of_posting_lists, common_docids = booleanSearchData
	
	for term, postingList in dict_of_posting_lists.items():
		df = postingList.df
		for posting in postingList.posting_list:
			tf = posting.tf
			tf_idf_weight_document[posting.docid][term] = get_tf_idf_weight(N, tf, df)

	# Calculate tf-idf weights of each term the query
	tf_idf_weight_query = defaultdict(float)

	tf_dict = defaultdict(int)
	query_size = len(query_words)
	for term in query_words:
		tf_dict[term] += 1
		
	for term, tf in tf_dict.items():
		tf_idf_weight_query[term] = tf/query_size


	tf_idf_query_magnitude = math.sqrt(sum([val*val for val in tf_idf_weight_query.values()]))

	for docid, data in tf_idf_weight_document.items():
		tf_idf_prod_weight_sum = 0
		tf_idf_doc_squared_sum = 0
		for term, tf_idf_weight in data.items():
			tf_idf_prod_weight_sum += tf_idf_weight * tf_idf_weight_query[term]
			tf_idf_doc_squared_sum += tf_idf_weight * tf_idf_weight

		tf_idf_doc_magnitude = math.sqrt(tf_idf_doc_squared_sum)
		normalizing_factor = tf_idf_query_magnitude * tf_idf_doc_magnitude
		cosine_similarity[docid] = tf_idf_prod_weight_sum / normalizing_factor

	return cosine_similarity


def tokenize(text):
	# Return list may contain duplicate tokens.
	tokenize_pattern = r"[a-zA-Z0-9]+"
	tokenizer = RegexpTokenizer(tokenize_pattern)
	return tokenizer.tokenize(text)


def generate_posting_lists(InvIndex_fh, MetaIndex, term):
	# Generates posting list(s) for the argument: term.
	# Merges posting lists if more than one exists for the same term.
	# Yields postings in a single posting list.
	
	boolDict = {"True": True, "False": False}

	# Gets the starting positions of term in inverted index file from meta index.
	try:
		record_positions = MetaIndex[term]
	except:
		# Exception raised if the term does not exist in inverted index
		raise Exception("Match Not Found")

	for record_position in sorted(record_positions):
		posting_list = PostingList()
		# Re-constructs posting list for each line specified by record_position
		InvIndex_fh.seek(record_position) # Moves file pointer to record position
		line = InvIndex_fh.readline() # Reads a single instance of (Term, PostingList(*))

		# Finds matches based on regex patterns.
		line_match = re.match(INVERTED_INDEX_LINE_PATTERN, line)
		posting_list_raw = line_match.group("PostingList")
		posting_list_match = re.match(INVERTED_INDEX_POSTINGLIST_PATTERN, posting_list_raw)

		yield_count = 0
		for posting_match in re.findall(INVERTED_INDEX_POSTING_PATTERN, posting_list_match.group("postings")):
			try:
				yield_count += 1
				docid = int(posting_match[0])
				tf =  int(posting_match[1])
				title =  boolDict[posting_match[2]]
				heading =  boolDict[posting_match[3]]
				bold =  boolDict[posting_match[4]]
				strong =  boolDict[posting_match[5]]
				italics =  boolDict[posting_match[6]]
				emphasized =  boolDict[posting_match[7]]
				termPositions =  [int(tp) for tp in posting_match[8].split(',')]
				fields = {"title": title, "heading": heading, "bold": bold, "strong":strong, "italics": italics, "emphasized":emphasized}

				# Create a new posting to add into the return posting list
				new_posting = Posting(docid, tf, fields)
				new_posting.termPositions = termPositions
				posting_list.append(new_posting)

				if yield_count >= RESULT_BATCH_SIZE:
					yield posting_list
					posting_list = PostingList()

			except AttributeError:
				pass

		yield posting_list


def generate_boolean_search_data(query_words):
	# {stemmed query word: posting list generator object}
	posting_list_generators = {query_word: generate_posting_lists(InvIndex_fh, MetaIndex, query_word) for query_word in query_words}

	while True:
		docid_list_backup = [] # Used when 2 or more query terms don't have common docids
		list_of_docid_lists = []
		dict_of_posting_lists = {}
		for query_word in sorted(query_words):
			try:
				posting_list = next(posting_list_generators[query_word])
			except:
				continue
			docid_list = [posting.docid for posting in posting_list.posting_list]
			list_of_docid_lists.append(docid_list)
			docid_list_backup.extend(docid_list)
			dict_of_posting_lists[query_word] = posting_list


		# Get list of common docids
		common_docids = set()
		for docids in list_of_docid_lists:
			if len(common_docids) == 0: common_docids = set(docids)
			else: common_docids = common_docids.intersection(set(docids))
		
		common_docids = sorted(list(common_docids))


		if len(common_docids) == 0:
			if len(docid_list_backup) > RESULT_BATCH_SIZE:
				common_docids = sorted(list(set(docid_list_backup)))[:RESULT_BATCH_SIZE]
			else:
				common_docids = sorted(list(set(docid_list_backup)))

		yield dict_of_posting_lists, common_docids


def get_search_results(DocIndex, terms, docids):
	search_results = []

	for docid in docids:
		doc_path = DocIndex[str(docid)][1]
		# The following json file contains ['url', 'content', 'encoding']
		# for the document referred to by docid
		with open(doc_path, 'r') as fh:
			json_object = json.load(fh)

		url, pageContent, encoding = json_object["url"], json_object["content"], json_object["encoding"]
		soup = BeautifulSoup(pageContent, 'lxml')
		text = soup.get_text()
		text_list = []

		if not DISPLAY_URLS_ONLY:
			for term in terms:
				if term in ' '.join(text_list).lower(): continue
				regex_pattern = r"(.{0,200}\W{1}"+term+r"\W{1}.{0,200})"
				try:
					new_display_text = re.search(regex_pattern, text, re.IGNORECASE).group(1)
				except AttributeError:
					continue
				text_list.append(new_display_text)

		search_results.append((url, text_list))

	return search_results

def display_search_results(search_results):
	for url, text_list in search_results:
		if DISPLAY_URLS_ONLY:
			print(url)

		else:
			print('='*80+'\n' + url)
			print(*text_list, sep="\n\t."*2+'\n')
			print('='*80)

def rank(booleanSearchData, cosine_similarity):
	ranked_docs = {}
	for _, posting_list in booleanSearchData[0].items():
		for posting in posting_list.posting_list:
			docid = posting.docid
			# Check if posting.docid exists in cosine_similarity.keys()
			try:
				cosine_similarity[docid]
			except KeyError:
				continue

			doc_score = 0
			# Add one to doc_score if term appears in title, heading, bold, strong, italics, or emphasized.
			if any(posting.fields.values()):
				doc_score += 1

			ranked_docs[docid] = cosine_similarity[docid] + doc_score

	return [docid for docid, score in sorted(ranked_docs.items(), key = lambda x: x[1], reverse=True)]




def main(InvIndex_fh, MetaIndex, DocIndex, TopResults = 5):
	# total number of documents in inverted index
	N = len(DocIndex)
	stemmer = snowball.SnowballStemmer('english')

	
	while True:
		query = input("\nEnter text to search or '-1' to exit: ")
		if query == '-1': break
		start_time = int(round(time.time() * 1000))
		
		query_words = tokenize(query)
		# Creates a new list of non-stopword query tokens
		filtered_query_words = [q for q in query_words if q not in STOP_WORDS]
		# Uses stop-word(s) for search if the query only contains stop words like "to be or not to be".
		if len(filtered_query_words) == 0: filtered_query_words = query_words

		stemmed_query_words = [stemmer.stem(query_word) for query_word in filtered_query_words]
		if len(stemmed_query_words) > QUERY_THRESHOLD: stemmed_query_words = stemmed_query_words[:QUERY_THRESHOLD]

		stemmed_query_word_set = set(stemmed_query_words)

		boolean_generator = generate_boolean_search_data(stemmed_query_word_set)

		load_next_set_of_data = False

		while True:
			cache_result = CACHE.get_result(query)
			if cache_result == None or load_next_set_of_data:
				booleanSearchData = next(boolean_generator)
				#search_results = get_search_results(DocIndex, filtered_query_words, booleanSearchData[1])

				cosine_similarity = compute_cosine_similarity(booleanSearchData, stemmed_query_words, N)

				if len(cosine_similarity) == 0: print("End of results")

				ranked_docids = rank(booleanSearchData, cosine_similarity)
				CACHE.add_result(query, ranked_docids)

			else:
				ranked_docids = cache_result

			end_time = int(round(time.time() * 1000))

			search_results = get_search_results(DocIndex, filtered_query_words, ranked_docids)
			display_search_results(search_results)
			print(f"Retrieval time: {end_time - start_time} milliseconds")

			cin = input(f"\nPress enter to see more results for {query}. \nEnter '0' to search something else\nEnter '-1' to quit\n")
			if cin == '0': break
			elif cin == '-1': raise SystemExit
			start_time = int(round(time.time() * 1000)) # Reset start time
			load_next_set_of_data = True



if __name__ == "__main__":
	if len(sys.argv) != 4:
		print(f"Expected 3 arguments (Inverted Index path, Meta Index path, Document Index path) received {len(sys.argv)-1} argument(s) instead")
		raise SystemExit

	InvIndexPath = sys.argv[1]
	MetaIndexPath = sys.argv[2]
	DocIndexPath = sys.argv[3]

	# Keep file handle open for reading contents from inverted index.
	# Inverted index could be too large to load into memory all at once. 
	InvIndex_fh = open(InvIndexPath, "r")
	# MetaIndex: JSON object with key = term and value = data offset
	# for each [term, postingList] in the inverted index file.
	with open(MetaIndexPath, 'r') as fh:
		MetaIndex = json.load(fh)
	# DocIndex: {key = docID: integer, value = (url: string, doc_path:string)}: JSON Object
	with open(DocIndexPath, 'r') as fh:
		DocIndex = json.load(fh)

	main(InvIndex_fh, MetaIndex, DocIndex)

	InvIndex_fh.close()
	