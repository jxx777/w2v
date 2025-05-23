def run_simple_queries(model) -> None:
    # 1. Retrieve the top most similar (topn) words using most_similar:
    query_word = "music"
    similar_words = model.wv.most_similar(query_word, topn=15)
    print(f"Most similar words for '{query_word}': {similar_words}")

    # 2. Compute similarity between two words:
    word1 = "food"
    word2 = "nutrition"
    similarity_score = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity_score:.2f}\n")

    # 3. Find the word that doesn't match in a list:
    words_list = ["fox", "rabbit", "motorcycle", "cat"]
    odd_word = model.wv.doesnt_match(words_list)
    print(f"Word that doesn't match in {words_list}: {odd_word}\n")

    # 4.1. Get the vector for a given word:
    query_for_vector = "technology"
    vector_for_query = model.wv.get_vector(query_for_vector)
    print(f"Vector for '{query_for_vector}':\n{vector_for_query}\n")

    # 4.2. Given above vector, find similar words using similar_by_vector:
    print(f"Similar words for the vector of '{query_for_vector}'")
    similar_by_vector = model.wv.similar_by_vector(vector_for_query)
    print(f"{similar_by_vector}\n")
