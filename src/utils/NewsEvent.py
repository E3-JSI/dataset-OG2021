import torch
import torch.nn.functional as f

from src.utils.LinearAlgebra import (
    get_intra_distances,
    get_centroid,
    update_centroid,
    get_min,
    get_max,
    get_avg,
    get_var,
)

from functools import reduce
import operator


class NewsEvent:
    """The class describing the event instance"""

    def __init__(self, articles=[]):
        # initialize the event properties
        self.articles = articles
        self.centroid = None
        self.c_norm = None
        self.named_entities = set()
        self.wiki_concepts = set()
        self.time_interval = None

        # update the event properties
        self._init_centroid()
        self._init_named_entities()
        # self._init_wiki_concepts()
        self._init_time_interval()

    # ==================================
    # Default Override Methods
    # ==================================

    def __repr__(self):
        return f"NewsEvent(\n  " f"n_articles={len(self.articles)},\n" ")"

    # ==================================
    # Class Methods
    # ==================================

    def add_article(self, article):

        # append the article
        self.articles.append(article)

        # update the event values
        self._update_centroid()
        self._update_named_entities()
        # self._update_wiki_concepts()
        self._update_time_interval()

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
            self.centroid = None
            self.c_norm = None
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

    def _init_wiki_concepts(self):
        if len(self.articles) == 0:
            # there are no articles
            self.wiki_concepts = set()
            return
        # get the article wikipedia concepts
        wiki = [a.get_wiki_concepts() for a in self.articles]
        self.wiki_concepts = reduce(operator.or_, wiki)

    def _init_time_interval(self):
        if len(self.articles) == 0:
            # there are no articles
            self.time_interval = None
            return
        # get the article times
        times = [a.time for a in self.articles]
        self.time_interval = {
            "min": get_min(times),
            "avg": get_avg(times),
            "max": get_max(times),
        }

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

    def _update_wiki_concepts(self):
        if len(self.articles) == 0:
            # there are no wikipedia concepts to extract
            self.wiki_concepts = set()
        elif len(self.articles) == 1:
            # there is only one article to extract wikipedia concepts from
            self.wiki_concepts = self.articles[0].get_wiki_concepts()
        else:
            # append the latest wikipedia concepts to the cluster
            wiki = self.articles[-1].get_wiki_concepts()
            self.wiki_concepts = self.wiki_concepts | wiki

    def _update_time_interval(self):
        if len(self.articles) != 0:
            times = [a.time for a in self.articles]
            self.time_interval = {
                "min": get_min(times),
                "avg": get_avg(times),
                "max": get_max(times),
            }

    # ==================================
    # Merge Methods
    # ==================================

    # TODO: implement merge methods

    # ==================================
    # Split Methods
    # ==================================

    # TODO: implement split methods
