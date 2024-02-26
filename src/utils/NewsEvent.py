import operator
from functools import reduce
from src.utils.LinearAlgebra import (
    get_intra_distances,
    get_centroid,
    update_centroid,
)

from src.utils.NewsEventBase import NewsEventBase


class NewsEvent(NewsEventBase):
    """The class describing the monolingual event instance"""

    def __init__(self, articles=[], use_ne=False):
        super().__init__(articles)

        # initialize the event properties
        self.use_ne = use_ne
        self.centroid = None
        self.c_norm = None
        self.time_interval = None
        # update the event properties
        self._init_centroid()
        if self.use_ne:
            self._init_named_entities()

    # ==================================
    # Class Methods
    # ==================================

    def add_article(self, article):
        super().add_article(article)

        # update the event values
        self._update_centroid()
        if self.use_ne:
            self._update_named_entities()

    def add_articles(self, articles):
        # append the articles
        for article in articles:
            self.add_article(article)

    def get_intra_distances(self):
        a_embeds = [a.get_content_embedding() for a in self.articles]
        # get intracluster distances
        intra_dist = get_intra_distances(a_embeds, self.centroid)
        # return the distances
        return intra_dist

    # ==================================
    # Initialization Methods
    # ==================================

    def _init_centroid(self):
        if len(self.articles) == 0:
            # there are no articles
            self.centroid, self.c_norm = None, 0
            return
        # get the centroid and its norm
        a_embeds = [a.get_content_embedding() for a in self.articles]
        self.centroid, self.c_norm = get_centroid(a_embeds)

    def _init_named_entities(self):
        if len(self.articles) == 0:
            # there are no articles
            self.named_entities = set()
            return
        # get the article named entities
        ne = [a.get_named_entities() for a in self.articles]
        self.named_entities = reduce(operator.or_, ne)

    # ==================================
    # Update Methods
    # ==================================

    def _update_centroid(self):
        if len(self.articles) == 0:
            # there are no articles
            self.centroid, self.c_norm = None, 0
        elif len(self.articles) == 1:
            # there is only one article in the cluster
            self.centroid, self.c_norm = get_centroid(
                [self.articles[0].get_content_embedding()]
            )
        else:
            # update the cluster centroid
            n_articles = len(self.articles) - 1
            a_embed = self.articles[-1].get_content_embedding()
            self.centroid, self.c_norm = update_centroid(
                self.centroid, self.c_norm, n_articles, a_embed
            )

    def _update_named_entities(self):
        if len(self.articles) == 0:
            # there are no named entities to extract
            self.named_entities = set()
        elif len(self.articles) == 1:
            # there is only one article to extract named entities from
            self.named_entities = self.articles[0].get_named_entities()
        else:
            # append the latest named entities to the cluster
            ne = self.articles[-1].get_named_entities()
            self.named_entities = self.named_entities | ne
