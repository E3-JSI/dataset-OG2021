import os
import re
import torch
import pathlib
import datetime
from typing import List

# import models
from src.models.MultilingualNER import MultilingualNER
from src.models.MultilingualLM import MultilingualLM
from src.utils.Wikifier import Wikifier

MODELS_PATH = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "models"
)

# ===============================================
# Helper Functions
# ===============================================

# format the strings
regex_whitespace = re.compile(r"(\s){1,}", re.IGNORECASE)
format_string = lambda x: re.sub(regex_whitespace, " ", x).strip()

# ===============================================
# Initialize Models
# ===============================================

model_type = "sbert"
pooling_type = "mean"
# initialize LM
embed_model = MultilingualLM(model_type=model_type, pooling_type=pooling_type).eval()

# initialize NER
checkpoint_path = f"{MODELS_PATH}/xlm-roberta-base-conll2003.ckpt"
#ner_model = MultilingualNER.load_from_checkpoint(checkpoint_path=checkpoint_path)

# initialize Wikifier
wikifier = Wikifier(user_key="cchuvmnmtiopyrekqoyupcmusiwtmk")


# ===============================================
# Define new Types
# ===============================================

from typing import TypedDict, Optional, Tuple, Set, List


class ArticleSource(TypedDict):
    """The class describing the article source attributes"""

    title: str


class Article(TypedDict):
    """The class describing the article attributes"""

    title: str
    body: str
    lang: str
    source: ArticleSource
    dateTime: str
    url: str
    uri: str
    eventUri: str
    concepts: List[str]


class ExperimentArticle(TypedDict):
    """The class describing the experiment article"""

    title: str
    source: str
    date: str
    lang: str
    cluster: str
    event_id: str


# ===============================================
# Define the News Article
# ===============================================


class NewsArticle:
    """The class describing the article instance"""

    title: str
    body: str
    lang: str
    source: str
    time: float
    url: str
    uri: str
    event_id: str
    concept: str
    cluster_id: int
    content_embedding: Optional[torch.Tensor]
    wiki_concepts: Optional[Set[str]]
    named_entities: Optional[Set[Tuple[str, str]]]

    # format="%Y-%m-%dT%H:%M:%SZ"
    def __init__(self, article: Article) -> None:
        self.title = format_string(article["title"])
        self.body = format_string(article["body"])
        self.source = article["source"]["title"]
        self.time = article["dateTime"].timestamp()
        self.lang = article["lang"]
        self.url = article["url"]
        self.uri = article["uri"]
        self.event_id = article["eventUri"]
        self.concept = article["concepts"]

        self.cluster_id = None

        # representation placeholders
        self.content_embedding = None
        self.named_entities = set()
        self.wiki_concepts = set()

    # ==================================
    # Default Override Methods
    # ==================================

    def __repr__(self) -> str:
        return (
            f"NewsArticle(\n  "
            f"title={self.title},\n  "
            f"time={self.time},\n  "
            f"body={self.body[0:1000]},\n"
            ")"
        )

    def __eq__(self, article: "NewsArticle") -> bool:
        return (
            self.title == article.title
            and self.body == article.body
            and self.lang == article.lang
            and self.source == article.source
            and self.time == article.time
            and self.url == article.url
        )

    def __ne__(self, article: "NewsArticle") -> bool:
        return not self == article

    def __ge__(self, article: "NewsArticle") -> bool:
        return self.time >= article.time

    def __gt__(self, article: "NewsArticle") -> bool:
        return self.time > article.time

    def __lt__(self, article: "NewsArticle") -> bool:
        return self.time < article.time

    def __le__(self, article: "NewsArticle") -> bool:
        return self.time <= article.time

    # ==================================
    # Class Methods
    # ==================================
    
    def get_text(self) -> str:
        return self.body
    
    def get_dfList_row(self):
        return [self.title, self.body, self.concept, self.get_time(), self.lang, self.event_id, self.url, self.source, self.cluster_id]
    
    def get_content_embedding(self) -> torch.Tensor:
        """Gets the content embedding
        Returns:
            embedding (torch.Tensor): The article content embedding.
        """
        if torch.is_tensor(self.content_embedding):
            # embedding is already available
            return self.content_embedding

        # prepare the content text
        content = f"{self.title} {self.body}"
        # get the content representation
        self.content_embedding = embed_model(content)[0]
        # return the content embedding
        return self.content_embedding

#    def get_named_entities(self) -> Set[Tuple[str, str]]:
#        """Gets the article named entities
#        Returns:
#            named_entities (Set[Tuple[str, str]]): The set of named entity
#                tuples, where the first element of the tuple is the named
#                entity and the second is the entity type.
#        """
#        if self.named_entities:
#            # entities are already available
#            return self.named_entities

#        # prepare the content text
#        content = f"{self.title} {self.body}"
#        # get the articles named entities
#        self.named_entities = ner_model(content)
#        self.named_entities = set(self.named_entities)
#        return self.named_entities

    def get_wiki_concepts(self):
        """Gets the article wikipedia concepts
        Returns:
            concepts (Set[str]): The set of Wikipedia concept titles.
        """

        def get_concept_title(concept: dict) -> str:
            return (
                concept["secTitle"]
                if "secTitle" in concept and concept["secTitle"]
                else concept["title"]
            )

        if self.wiki_concepts:
            # concepts are already available
            return self.wiki_concepts

        # prepare the content text
        content = f"{self.title} {self.body}"
        # get the wikipedia concepts from Wikifier
        self.wiki_concepts = wikifier.wikify(content)
        self.wiki_concepts = set([get_concept_title(c) for c in self.wiki_concepts])
        return self.wiki_concepts

    def get_time(self) -> datetime.datetime:
        """Gets the article time in a readable format
        Returns:
            time (datetime.datetime): The time when the article
                was published.
        """
        return datetime.datetime.fromtimestamp(self.time)
