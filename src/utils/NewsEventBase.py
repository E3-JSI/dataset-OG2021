import torch
from src.utils.LinearAlgebra import (
    get_min,
    get_max,
    get_avg,
)


class NewsEventBase:
    """The class describing the monolingual event instance"""

    def __init__(self, articles=[]):
        # initialize the event properties
        self.articles = articles
        self.time_interval = None

        # update the event properties
        self._init_time_interval()

    # ==================================
    # Default Override Methods
    # ==================================

    def __repr__(self):
        return f"NewsEvent(\n  " f"n_articles={len(self.articles)},\n" ")"

    @property
    def min_time(self):
        return self.get_time(metric="min")

    @property
    def avg_time(self):
        return self.get_time(metric="avg")

    @property
    def max_time(self):
        return self.get_time(metric="max")

    @property
    def cluster_id(self):
        return self.articles[0].cluster_id if len(self.articles) > 0 else None

    @property
    def lang(self):
        return self.articles[0].lang if len(self.articles) > 0 else None

    # ==================================
    # Class Methods
    # ==================================

    def add_article(self, article):
        # append the article
        self.articles.append(article)

        # update the event values
        self._update_time_interval()

    def add_articles(self, articles):
        self.articles.extend(articles)

        # update the event values
        self._update_time_interval()

    def get_article_embeddings(self):
        return torch.stack(
            [article.get_content_embedding() for article in self.articles]
        ).unsqueeze(0)

    def get_time(self, metric="avg"):
        if len(self.articles) == 0:
            return None
        return self.time_interval[metric]

    # ==================================
    # Initialization Methods
    # ==================================

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
