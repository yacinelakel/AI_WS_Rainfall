from conllu import parse


def read_data(path, ret_tokens=True):
    """
    Reading a text corpus from a CoNLL-U file format downloaded from Universal Dependencies

    :param path: the path to the file
    :param ret_tokens: whether to return the tokens of the extracted corpus
    :return: TokenList object containing the metadata and optional  list of all corpus tokens
    """

    # Example: Reading a CoNLL-U file
    with open(path, "r", encoding="utf-8") as file:
        conllu_data = file.read()
    # Parsing CoNLL-U data
    parsed_data = parse(conllu_data)

    # Extracting tokens for NLTK processing
    tokens = [token["form"] for sentence in parsed_data for token in sentence]

    if ret_tokens:
        return parsed_data, tokens
    else:
        return parsed_data, None
