import xxhash
from tqdm import tqdm
from collections import defaultdict

def parse_document(document_file, title_file):
    """
    Parses the input file 'documents'
    - Parses the first line which has three integers: n, k, and q
    - Parses the following lines of n documents, one on each line

    Parses the input file 'title'
    - Parses the following 'n' lines that has one title to correspond to each document

    :param document_file: the filename of the document
    :param title_file: the filename of the title

    :return: n (the number of documents), k (shingle size), q (number of disctinct similar pairs to output), 
    :return: the documents, the corresponding titles
    """
    with open(document_file, 'r') as file:
        n, k, q = map(int, file.readline().strip().split())

        documents = []
        for _ in range(n):
            document = file.readline().strip()
            documents.append(document)
    
    with open(title_file, 'r') as file:
        titles = []
        for _ in range(n):
            title = file.readline().strip()
            titles.append(title)

    return n, k, q, titles, documents

def k_shingle(k, document):
    """
    Generates all unique substrings (shingles) of length k

    :param k: the length of the substring (shingle)
    :param document: the input document / wikipedia article

    :return: set of unique shingles
    """
    unique_k_shingle = set()
    for index in range(len(document)-k+1):
        unique_k_shingle.add(document[index:index+k])
    return unique_k_shingle

def sim(document_set1: set, document_set2: set):
    """
    computes the Jaccard similarity of two documents
    - the ratio of the intersection of the two sets to their union

    :params document_set1: the first set
    :params document_set2: the second set

    :returns: a float measuring the ratio between their intersection and union
    """
    intersect = document_set1 & document_set2
    return 1.0 * len(intersect) / (len(document_set1) + len(document_set2) - len(intersect))

def min_hash_compute(document_set, hash_funcs):
    """
    Generates the minhash for a single document_set (k-shingle) for every single hash function
    
    :params document_set: a set of unique substrings (shingles) of length k for a document
    :params hash_funcs: a list of hash functions

    :returns: a list of min_hash for a document_set (k-shingle)
    """
    min_hashes = [float('inf')] * len(hash_funcs)
    for i, hash_func in enumerate(hash_funcs):
        for element in document_set:
            hash_value = hash_func(element)
            min_hashes[i] = min(min_hashes[i], hash_value)
    
    return min_hashes

def lsh(document_min_hashes, num_bands, num_rows, threshold):
    """
    Perform Locality Sensitive Hashing for documents

    :param document_min_hashes: a dictionary of {index: list of min hashes}
    :param num_bands: the number of bands to split the min hashes into
    :param num_rows: the number of rows per band
    :param threshold: the approx max number of unique similar document pairs

    :return: a list of tuples, each tuple is a similar document pair
    """
    
    doc_pairs = set()

    for band_idx in range(num_bands):
        # for each band, add the documents with the same hashes into the same bucket
        buckets = defaultdict(list)
        for doc_index, doc_min_hash in document_min_hashes.items():
            min_hash_tuple = tuple(doc_min_hash[band_idx*num_rows: (band_idx+1)*num_rows])
            buckets[min_hash_tuple].append(doc_index)
            
        # add all pairs in the same bucket to C
        for bucket in buckets.values():
            for i in range(len(bucket)):
                for j in range(i+1, len(bucket)):
                    # sorted to ensure that (i, j) and (j, i) are not both in doc_pairs
                    doc_pairs.add(tuple(sorted([bucket[i], bucket[j]])))

            if len(doc_pairs) >= threshold:
                return doc_pairs

        print(f"iteration {band_idx+1} done")
    
    return doc_pairs

def find_similarity(document_pairs, indices_to_documents, k):
    """
    Calculate Jaccard similarity for document pairs
    
    :params document_pairs: list of tuples containing pairs of document titles
    :params indices_to_documents: dictionary mapping document indices to document text
    :params k: size of the k-shingles
    
    :returns: list of tuples (doc1_index, doc2_index, Jaccard similarity_score)
    """
    results = []
    
    for doc1_index, doc2_index in document_pairs:
        # generate document sets, based on k_shingle
        doc1 = indices_to_documents[doc1_index]
        doc2 = indices_to_documents[doc2_index]

        doc_set1 = k_shingle(k, doc1)
        doc_set2 = k_shingle(k, doc2)

        intersection_size = len(doc_set1 & doc_set2)
        
        # Calculate Jaccard similarity
        similarity = intersection_size / (len(doc_set1) + len(doc_set2) - intersection_size)
        
        results.append((doc1_index, doc2_index, similarity))
    
    return results

# Example usage
if __name__ == "__main__":
    n, k, q, titles, documents = parse_document("documents", "wiki.titles")

    # Step 1: Pick b and r, (and t)
    num_bands = 20
    num_rows = 5
    threshold = n // 2
    
    hash_funcs = [lambda x, idx=idx: xxhash.xxh32(x, seed=42+idx).intdigest() for idx in range(num_bands*num_rows)]

    # Step 2: Run (br) minHash per document
    document_min_hashes = dict()
    for index, document in tqdm(enumerate(documents)):
        # 1. generate k-shingle
        document_set = k_shingle(k, document)
        # 2. generate min-hash
        document_min_hashes[index+1] = min_hash_compute(document_set, hash_funcs)
    print('finished creating min hashes')
    
    # Step 3: Apply LSH, to find "t" similar documents
    doc_pairs = lsh(document_min_hashes, num_bands, num_rows, threshold=threshold)
    
    # Step 4: Figure out similarity between all "t" document pairs
    indices_to_documents = {}
    for index, document in enumerate(documents):
        indices_to_documents[index+1] = document
    sorted_doc_pairs = find_similarity(doc_pairs, indices_to_documents, k)
    sorted_doc_pairs.sort(key=lambda x: x[2], reverse=True)

    # Step 5: Save the highest "q" similar document pairs
    with open('lhs_ans', 'w') as file:
        for pair in sorted_doc_pairs[:q]:
            file.write(f"{pair[0], pair[1]}\n")
    
    with open('lhs_similarity', 'w') as file:
        for pair in sorted_doc_pairs[:q]:
            file.write(f"{pair[0], pair[1], pair[2]}\n")
    print('Done!')