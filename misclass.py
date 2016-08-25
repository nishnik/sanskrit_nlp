# writes to misclassified_+FILE, as [correct][predicted]:first_word,second_word

import json

FILE = "test.txt"

dict_ = {}
with open(FILE) as json_data:
    dict_ = json.load(json_data)

misclassified = {}
misclassified["A"] = {}
misclassified["B"] = {}
misclassified["D"] = {}
misclassified["T"] = {}

misclassified["A"]["B"] = []
misclassified["A"]["D"] = []
misclassified["A"]["T"] = []

misclassified["B"]["A"] = []
misclassified["B"]["D"] = []
misclassified["B"]["T"] = []

misclassified["D"]["B"] = []
misclassified["D"]["A"] = []
misclassified["D"]["T"] = []

misclassified["T"]["B"] = []
misclassified["T"]["D"] = []
misclassified["T"]["A"] = []

for a in dict_:
	if (dict_[a]["predicted"] != dict_[a]["actual"]):
		misclassified[dict_[a]["actual"]][dict_[a]["predicted"]].append(a)


with open('misclassified_' + FILE, 'w') as outfile:
     json.dump(misclassified, outfile, indent = 4, ensure_ascii=False)

