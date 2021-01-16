# Author: Shuvam Raj Satyal

import sys
import pickle
from datetime import datetime

CACHE_SIZE = 100000000 # 100 MB

class Search_Cache:
	def __init__(self):
		try:
			with open("../CACHE.pkl", "rb") as fh:
				self.__search_results = pickle.load(fh)
		except FileNotFoundError:
			self.__search_results = {}

	def add_result(self, query, result):
		try:
			self.__search_results[query]["Search Date-Time"] = datetime.now()
			self.__search_results[query]["Search Count"] += 1
			# Results are not updated assuming that initial results are higher ranked

		except KeyError:
			self.__search_results[query] = {"Search Date-Time": datetime.now(), "Search Count": 1, "Results": result}

		if sys.getsizeof(self.__search_results) > CACHE_SIZE: self.remove_low_priority_query()

		with open("../CACHE.pkl", "wb") as fh:
			pickle.dump(self.__search_results, fh)


	def get_result(self, query):
		try:
			return self.__search_results[query]["Results"]
		except KeyError:
			return None

	def remove_low_priority_query(self):
		results = sorted(self.__search_results.items(), key=lambda x: x[1]["Search Date-Time"])
		while sys.getsizeof(self.__search_results) > CACHE_SIZE:
			if len(results) > 1:
				result1 = results[0]
				result2 = results[1]
				if result1[1]["Search Count"] <= result2[1]["Search Count"]:
					results.pop(0)
					del self.__search_results[result1[0]]
				else:
					results.pop(1)
					del self.__search_results[result2[0]]
			else: break