from classifier import keyword_detection, sentence_classifier

TARGET_KEYWORDS = ['SHIPPER','CONSIGNEE', 'NOTIFY', 'ALSO_NOTIFY', 'POR', 'POL', 'POD', 'DEL', 'VESSEL']
# Test keyword
sentence = 'shipper/exporter'
keyword = keyword_detection(sentence)
print('keyword --->', keyword)

sentence = 'pre-carriage by *'
keyword = keyword_detection(sentence)
print('keyword --->', keyword)

sentence = 'shipper exporter'
keyword = keyword_detection(sentence)
print('keyword --->', keyword)

# Test sentence classifiation, input value sentence -> output class
sentence = "SEATTLE, WA"
result = sentence_classifier(sentence)
print(result)