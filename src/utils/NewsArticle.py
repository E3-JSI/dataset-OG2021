import os
import re
import torch
import pathlib
import datetime
from typing import Set, Tuple, List, Union

# import models
from src.models.MultilingualLM import MultilingualLM
from src.models.MultilingualNER import MultilingualNER
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

# initialize LM
embed_model = MultilingualLM(
    model_type="sbert", pooling_type="mean", use_gpu=True
).eval()

# initialize NER
ner_model = MultilingualNER(use_gpu=True).eval()

# initialize Wikifier
wikifier = Wikifier(user_key="cchuvmnmtiopyrekqoyupcmusiwtmk")


# ===============================================
# Define new Types
# ===============================================

from typing import TypedDict, Optional, List


class ArticleSource(TypedDict):
    """The class describing the article source attributes"""

    title: str


class Article(TypedDict):
    """The class describing the article attributes"""

    title: str
    body: str
    lang: str
    source: Union[ArticleSource, str]
    dateTime: str
    url: str
    uri: str
    eventUri: str
    concepts: List[str]
    clusterId: str
    namedEntities: Union[List[Tuple[str, str]], None]
    wikiConcepts: Union[List[str], None]


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
    concepts: str
    cluster_id: str
    content_embedding: Optional[torch.Tensor]

    # format="%Y-%m-%dT%H:%M:%SZ"
    def __init__(self, article: Article) -> None:
        self.title = format_string(article["title"])
        self.body = format_string(article["body"])
        self.source = (
            article["source"]["title"]
            if isinstance(article["source"], dict)
            else article["source"]
        )
        self.time = article["dateTime"].timestamp()
        self.lang = article["lang"]
        self.url = article["url"]
        self.uri = article["uri"]
        self.event_id = article["eventUri"]
        self.concepts = article["concepts"]

        self.cluster_id = (
            article["clusterId"]
            if "clusterId" in article and article["clusterId"] is not None
            else None
        )

        # representation placeholders
        self.content_embedding = None
        self.named_entities = (
            set(article["namedEntities"])
            if "namedEntities" in article and article["namedEntities"] is not None
            else None
        )
        self.wiki_concepts = (
            set(article["wikiConcepts"])
            if "wikiConcepts" in article and article["wikiConcepts"] is not None
            else None
        )

    # ==================================
    # Default Override Methods
    # ==================================

    def __repr__(self) -> str:
        return (
            f"NewsArticle(\n  "
            f"title={self.title},\n  "
            f"body={self.body[0:1000]},\n"
            f"source={self.source},\n  "
            f"time={self.time},\n  "
            f"lang={self.lang},\n  "
            f"url={self.url},\n  "
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

    def to_array(self):
        return [
            self.title,
            self.body,
            self.lang,
            self.source,
            self.get_time(),
            self.url,
            self.uri,
            self.event_id,
            self.concepts,
            self.cluster_id,
            list(self.named_entities) if self.named_entities else None,
            list(self.wiki_concepts) if self.wiki_concepts else None,
        ]

    def get_text(self) -> str:
        return f"{self.title} {self.body}"

    def get_content_embedding(self) -> torch.Tensor:
        """Gets the content embedding
        Returns:
            embedding (torch.Tensor): The article content embedding.
        """
        if torch.is_tensor(self.content_embedding):
            # embedding is already available
            return self.content_embedding

        # get the content representation
        self.content_embedding = embed_model(self.get_text())[0]
        # return the content embedding
        return self.content_embedding

    def get_named_entities(self) -> Set[Tuple[str, str]]:
        """Gets the article named entities
        Returns:
            named_entities (Set[Tuple[str, str]]): The set of named entity
                tuples, where the first element of the tuple is the named
                entity and the second is the entity type.
        """
        if self.named_entities:
            # entities are already available
            return self.named_entities

        # get the articles named entities
        self.named_entities = ner_model(self.get_text())
        self.named_entities = set(
            [(ne["word"], ne["entity_group"]) for ne in self.named_entities]
        )
        return self.named_entities

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

        # get the wikipedia concepts from Wikifier
        self.wiki_concepts = wikifier.wikify(self.get_text())
        self.wiki_concepts = set([get_concept_title(c) for c in self.wiki_concepts])
        return self.wiki_concepts

    def get_time(self) -> datetime.datetime:
        """Gets the article time in a readable format
        Returns:
            time (datetime.datetime): The time when the article
                was published.
        """
        return datetime.datetime.fromtimestamp(self.time)
