# Author: Shuvam Raj Satyal

from collections import defaultdict
from shutil import copyfile, rmtree
import re
from Inverted_Index import *

# The optimal document batch size depends on hardware, OS, programming language, data structures used, etc
DOCUMENT_BATCH_SIZE = 18465
# Block size for binary merge
MERGE_BLOCK_SIZE = 10 * 1000000 # 10 MB

# Patterns to re-construct posting lists from text files during merge
INVERTED_INDEX_LINE_PATTERN = r"Term:(?P<term>\w+),PostingList:\[(?P<PostingList>.+)\]\n"
INVERTED_INDEX_POSTINGLIST_PATTERN = r"df:(?P<df>\d+),Postings:\[(?P<postings>.+)\]"
INVERTED_INDEX_POSTING_PATTERN = r"docid:(\d+),tf:(\d+),fields:\[title:(\w+),heading:(\w+),bold:(\w+),strong:(\w+),italics:(\w+),emphasized:(\w+)\],termPositions:\[([0-9,]+)\]"


def generate_document_paths(document_paths):
	# yields a batch of document paths with batch size = DOCUMENT_BATCH_SIZE
	document_count = 0
	batch = []

	for document_path in document_paths:
		batch.append(document_path)
		document_count +=1
		if document_count == DOCUMENT_BATCH_SIZE:
			yield batch
			document_count = 0
			batch = []

	if len(batch) != 0:
		yield batch



def BuildPartialInvertedIndexes(document_paths, storageDirPath, partialIndexesDirPath):
	path_gen = generate_document_paths(document_paths) # Generator oject that yields subset of document paths based on DOCUMENT_BATCH_SIZE
	max_document_index = 0
	invertedIndex_count = 1 # Used for naming partial Inverted Indexes on disk

	while True:
		try:
			# Gets a subset of document paths from the generator
			batch = next(path_gen)

			# Builds the partial inverted Index and returns Document Index along with partial Inverted Index
			DocumentIndex, InvertedIndex = BuildInvertedIndex(batch)

			# Sort inverted index and write to disk
			with open(os.path.join(partialIndexesDirPath, f"InvIndex_{invertedIndex_count}.txt"), 'w') as fh:
				for term, posting_list in sorted(InvertedIndex.items()):
					postings = []
					for posting in posting_list.posting_list:
						posting_term_positions = ','.join([str(tp) for tp in posting.termPositions])
						posting_string = f"Posting(docid:{posting.docid+max_document_index},tf:{posting.tf},"\
						f"fields:[title:{posting.fields['title']},heading:{posting.fields['heading']},"\
						f"bold:{posting.fields['bold']},strong:{posting.fields['strong']},"\
						f"italics:{posting.fields['italics']},emphasized:{posting.fields['emphasized']}],"\
						f"termPositions:[{posting_term_positions}])"
						postings.append(posting_string)

					write_line = f"Term:{term},PostingList:[df:{posting_list.df},Postings:[{','.join(postings)}]]\n"
					fh.write(write_line)
			
			invertedIndex_count += 1

			# Update index of each document in DocumentIndex based on the max_document_index from previous batch
			DocumentIndex = {k + max_document_index : v for k,v in DocumentIndex.items()}

			# Set max_document_index to the highest document index in current batch
			max_document_index = max(DocumentIndex.keys())

			# Append new document indexes to existing DocumentIndex.json
			try:
				with open(os.path.join(storageDirPath, "DocIndex.json"), 'r') as fh:
					prev_doc_indexes = json.load(fh)

				# Merge the previous document indexes with current ones and store into disk
				with open(os.path.join(storageDirPath, "DocIndex.json"), 'w') as fh:
					json.dump({**prev_doc_indexes, **DocumentIndex}, fh, indent=3)

			except FileNotFoundError:
				with open(os.path.join(storageDirPath, "DocIndex.json"), 'w') as fh:
					json.dump(DocumentIndex, fh, indent=3)

		except StopIteration:
			break

def get_posting_list_from_txt_file(line):
	boolDict = {"True": True, "False": False}
	posting_list = PostingList()

	line_match = re.match(INVERTED_INDEX_LINE_PATTERN, line)
	term = line_match.group("term")
	posting_list_raw = line_match.group("PostingList")
	posting_list_match = re.match(INVERTED_INDEX_POSTINGLIST_PATTERN, posting_list_raw)

	for posting_match in re.findall(INVERTED_INDEX_POSTING_PATTERN, posting_list_match.group("postings")):
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

	return (term, posting_list)


def BinaryMerge(partialIndexesDirPath, mergedIndexesDirPath):
	# Merges Posting lists from two files at a time

	# mergeQueue initially contains paths of partial indexes
	mergeQueue = sorted([os.path.join(partialIndexesDirPath, f) for f in os.listdir(partialIndexesDirPath) if os.path.isfile(os.path.join(partialIndexesDirPath, f)) and f != ".DS_Store"])
	output_file_count = 0

	while len(mergeQueue) > 1:
		output_file_count += 1
		fh1 = open(mergeQueue.pop(0), "r")
		fh2 = open(mergeQueue.pop(0), "r")

		mergedIndex_path = os.path.join(mergedIndexesDirPath, f"MergedIndex_{output_file_count}.txt")
		mergeQueue.append(mergedIndex_path)
		output_fh = open(mergedIndex_path, 'w')

		block_1 = {}
		block_2 = {}
		output_dict = defaultdict(PostingList)
		EOF_1 = False
		EOF_2 = False

		print(f"Merging {fh1.name} and {fh2.name} into {output_fh.name}")

		while True:
			try:
				if EOF_1: raise EOFError
				while sys.getsizeof(block_1) < MERGE_BLOCK_SIZE:
					# Load a block of data of size = MERGE_BLOCK_SIZE
					line = fh1.readline()
					data = get_posting_list_from_txt_file(line)
					# {key = term, value = PostingList}
					block_1[data[0]] = data[1]

			except (EOFError, AttributeError):
				EOF_1 = True

			try:
				if EOF_2: raise EOFError
				while sys.getsizeof(block_2) < MERGE_BLOCK_SIZE:
					# Load a block of data of size = MERGE_BLOCK_SIZE
					line = fh2.readline()
					data = get_posting_list_from_txt_file(line)
					# {key = term, value = PostingList}
					block_2[data[0]] = data[1]
					
			except (EOFError, AttributeError):
				EOF_2 = True


			# Perform merge opeation
			for key in set().union(block_1, block_2):
				for dic in [block_1, block_2]:
					if key in dic:
						output_dict[key].posting_list += dic[key].posting_list
						output_dict[key].df += dic[key].df
						output_dict[key].posting_list.sort(key=lambda x: x.docid)


			# Sort inverted index and write to disk
			for term, posting_list in sorted(output_dict.items()):
				postings = []
				for posting in posting_list.posting_list:
					posting_term_positions = ','.join([str(tp) for tp in posting.termPositions])
					posting_string = f"Posting(docid:{posting.docid},tf:{posting.tf},"\
					f"fields:[title:{posting.fields['title']},heading:{posting.fields['heading']},"\
					f"bold:{posting.fields['bold']},strong:{posting.fields['strong']},"\
					f"italics:{posting.fields['italics']},emphasized:{posting.fields['emphasized']}],"\
					f"termPositions:[{posting_term_positions}])"
					postings.append(posting_string)

				write_line = f"Term:{term},PostingList:[df:{posting_list.df},Postings:[{','.join(postings)}]]\n"
				output_fh.write(write_line)

			block_1 = {}
			block_2 = {}
			output_dict = defaultdict(PostingList)

			# Exit while loop if EOF reached for both files
			if EOF_1 and EOF_2: break
 
		# Close file handles
		fh1.close()
		fh2.close()
		output_fh.close()



def extractFinalIndex(inv_index_name, storage_dir_path, partial_indexes_dir_path, merged_indexes_dir_path, delete_sub_indexes=False):
	# Extract the largest merged index and save it in the folder specified by storage_dir_path/inv_index_name
	largest_file = None
	largest_file_size = 0

	merged_index_paths = [os.path.join(merged_indexes_dir_path, f) for f in os.listdir(merged_indexes_dir_path) if os.path.isfile(os.path.join(merged_indexes_dir_path, f))]

	for path in merged_index_paths:
		file_size = os.path.getsize(path)
		if file_size > largest_file_size:
			largest_file_size = file_size
			largest_file = path

	copyfile(largest_file, os.path.join(storage_dir_path, inv_index_name))

	if delete_sub_indexes:
		rmtree(partial_indexes_dir_path)
		rmtree(merged_indexes_dir_path)



def BuildMetaIndex(meta_index_name, inv_index_name, storage_dir_path):
	meta_index = defaultdict(list)
	meta_index_path = os.path.join(storage_dir_path, meta_index_name)
	inv_index_path = os.path.join(storage_dir_path, inv_index_name)

	with open(inv_index_path, 'r') as fh:
		while True:
			try:
				cur_pos = fh.tell()
				line = fh.readline()
				term = re.match(r"^Term:(\w+).+$", line).group(1)
				meta_index[term].append(cur_pos)
			except (EOFError,AttributeError):
				break
				
	with open(meta_index_path, 'w') as fh:
		json.dump(meta_index, fh, indent=2)


if __name__ == "__main__":
	partial_indexes_dir_name = 'Partial_Indexes'
	merged_indexes_dir_name = 'Merged_Indexes'
	inv_index_name = 'InvIndex.txt'
	meta_index_name = "MetaIndex.json"


	if len(sys.argv) != 3:
		print(f"Expected 2 arguments(File Paths) received {len(sys.argv)-1} argument(s) instead")
		raise SystemExit

	corpus_path = sys.argv[1]
	storage_dir_path = sys.argv[2]

	partial_indexes_dir_path = os.path.join(storage_dir_path, partial_indexes_dir_name)
	if not os.path.exists(partial_indexes_dir_path): os.makedirs(partial_indexes_dir_path)

	document_paths = get_document_paths(corpus_path)
	BuildPartialInvertedIndexes(document_paths, storage_dir_path, partial_indexes_dir_path)

	merged_indexes_dir_path = os.path.join(storage_dir_path, merged_indexes_dir_name)
	if not os.path.exists(merged_indexes_dir_path): os.makedirs(merged_indexes_dir_path)

	BinaryMerge(partial_indexes_dir_path, merged_indexes_dir_path)
	extractFinalIndex(inv_index_name, storage_dir_path, partial_indexes_dir_path, merged_indexes_dir_path, delete_sub_indexes=True)
	BuildMetaIndex(meta_index_name, inv_index_name, storage_dir_path)