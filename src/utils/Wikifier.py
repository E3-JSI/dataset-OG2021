import re
import urllib.parse, urllib.request, json


class Wikifier:
    def __init__(
        self, user_key, max_length=20000, url="http://www.wikifier.org/annotate-article"
    ):
        self.user_key = user_key
        self.max_length = max_length
        self.url = url

    def prepare_text(self, text):
        """Prepares the text to be processed (splitting into chunks)
        Args:
            text (string): The text to be processed.
        Return:
            (string, number)[]: The list of string, weight pairs.
        """
        chunks = []
        current_index = 0

        while len(text) > current_index:
            # split text into chunks
            chunk = text[current_index : current_index + self.max_length]
            # if the chunk is empty, break the cycle
            if len(chunk) == 0:
                break

            elif len(chunk) == self.max_length:
                # find the optimal position to cut the text
                cut_index = None
                eos_positions = [m.end(0) for m in re.finditer(r"[\.?!]", chunk)]
                if len(eos_positions) > 0:
                    # if there is an end to the sentence
                    cut_index = eos_positions[-1]
                if not cut_index:
                    # find where is the last space
                    last_space_position = chunk.rfind(" ")
                    if last_space_position != -1:
                        cut_index = last_space_position
                if not cut_index:
                    # make no special cutting
                    cut_index = len(chunk)

                # cut the chunk in the optiomal position
                chunk = chunk[0:cut_index]
                current_index = current_index + cut_index
            else:
                current_index = current_index + len(chunk)

            weight = len(chunk) / len(text)
            chunks.append((chunk, weight))
        return chunks

    def get_wiki_concepts(self, text, lang="auto", threshold=0.8):
        """Retrieves wikipedia concepts relevant to the given text
        Args:
            text (string): The text to be wikified.
            lang (string): The language of the text. If "auto" it will
                automatically try to detect the language. Default: "auto".
            theshold (number): The threshold for pruning the annotations on the
                basis of their pagerank score. Default: 0.8.
        Returns:
            The wikipedia concepts relevant to the given text.
        """
        # prepare the request values
        data = urllib.parse.urlencode(
            [
                ("text", text),
                ("lang", lang),
                ("userKey", self.user_key),
                ("pageRankSqThreshold", "%g" % threshold),
                ("applyPageRankSqThreshold", "true"),
                ("nTopDfValuesToIgnore", "200"),
                ("nWordsToIgnoreFromList", "200"),
                ("wikiDataClasses", "false"),
                ("wikiDataClassIds", "false"),
                ("support", "false"),
                ("ranges", "false"),
                ("minLinkFrequency", "2"),
                ("includeCosines", "true"),
                ("maxMentionEntropy", "3"),
            ]
        )

        # prepare the request object
        req = urllib.request.Request(self.url, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))

        # output the annotations
        return response["annotations"]

    def weight_concept(self, concept, weight):
        """Weight the cosine and pageRank values
        Args:
            concept (dict): The wikipedia concept to be weighted.
            weight (number): The weight.
        Returns:
            The weighted wikipedia concept.
        """
        concept["cosine"] = concept["cosine"] * weight
        concept["pageRank"] = concept["pageRank"] * weight
        # adding possible missing attributes
        concept["secTitle"] = concept["secTitle"] if "secTitle" in concept else None
        concept["secLang"] = concept["secLang"] if "secLang" in concept else None
        concept["secUrl"] = concept["secUrl"] if "secUrl" in concept else None
        return concept

    def merge_concepts(self, wiki_chunks):
        """Merge the list of wikipedia concept chunks
        Args:
            wiki_chunks (dict[][]): The list of wikipedia concept chunk responses.
        Returns:
            List of merged wikipedia concepts.
        """
        mapping = {}
        for wiki_chunk in wiki_chunks:
            for concept in wiki_chunk:
                if concept["url"] in mapping:
                    mapping[concept["url"]]["cosine"] += concept["cosine"]
                    mapping[concept["url"]]["pageRank"] += concept["pageRank"]
                    mapping[concept["url"]]["supportLen"] += concept["supportLen"]
                else:
                    mapping[concept["url"]] = concept

        return list(mapping.values())

    def wikify(self, text):
        """Gets the wikipedia concepts for the whole text
        Args:
            text (string): The text to be processed.
        """
        wiki_chunks = []
        for chunk, weight in self.prepare_text(text):
            # get the wikipedia concepts for the given chunk and weight the concepts
            wiki_concepts = [
                self.weight_concept(concept, weight)
                for concept in self.get_wiki_concepts(chunk)
            ]
            wiki_chunks.append(wiki_concepts)

        # merge the wikipedia concepts
        merged_concepts = self.merge_concepts(wiki_chunks)
        # return the wikipedia concepts
        return merged_concepts
