import torch
from src.models.PairBERT import PairBERT
from src.utils.NewsEvent import NewsEvent
from src.utils.NewsArticle import NewsArticle
from src.utils.LinearAlgebra import cosine_similarity, jaccard_index

from typing import List

# ===============================================
# Define constants
# ===============================================

# one day in seconds
ONE_DAY = 86400

# ===============================================
# Define the News Event Monitor
# ===============================================


class NewsEventMonitor:
    active_events: List[NewsEvent]
    past_events: List[NewsEvent]
    sim_threshold: float
    time_threshold: int
    time_compare: str
    compare_model: PairBERT

    def __init__(
        self,
        sim_threshold: float = 0.5,
        time_threshold_in_days: int = 1,
        time_compare_stat: str = "avg",
        compare_threshold: float = 0.7,
        compare_model: PairBERT = None,
        compare_ne: bool = True,
    ) -> None:
        self.active_events = []
        self.past_events = []
        self.sim_threshold = sim_threshold
        self.time_threshold = time_threshold_in_days * ONE_DAY
        self.time_compare = time_compare_stat
        self.compare_named_entities = compare_ne
        self.compare_threshold = compare_threshold
        self.compare_model = compare_model

    # ==================================
    # Default Override Methods
    # ==================================

    # ==================================
    # Class Methods
    # ==================================

    def update(self, article: NewsArticle, device=None):
        """Update the events with the new article"""

        # get the events of the specific language
        lang_active_events = [
            event for event in self.active_events if event.lang == article.lang
        ]

        if len(lang_active_events) == 0:
            # create a new news event cluster
            event = NewsEvent(articles=[article], use_ne=self.compare_named_entities)
            self.active_events.append(event)
            return

        # calculate the similarity of the article to the events
        sims = torch.Tensor(
            [
                cosine_similarity(event.centroid, article.get_content_embedding())
                for event in lang_active_events
            ]
        )
        # get the sorted indices of the most similar events
        sort_index = torch.argsort(sims, descending=True)

        idx = 0
        assigned_to_event = False
        while sims[sort_index[idx]] > self.sim_threshold:
            # get the next closest news event
            event = lang_active_events[sort_index[idx]]

            # check if the article happened at an approximate
            # same time as the rest of the event articles
            time_diff = self.__absolute_difference(
                article.time, event.time_interval[self.time_compare]
            )

            # precalculate if the event and articles have similar entities
            has_similar_entities = (
                self.__has_similar_entities(
                    event.named_entities, article.get_named_entities()
                )
                if self.compare_named_entities
                else True
            )

            # classify if the article is similar enough to the event
            compare_score = self.compare_model(
                [article.get_text()], [event.articles[0].get_text()], device
            )
            if (
                has_similar_entities
                and compare_score > self.compare_threshold
                and time_diff <= self.time_threshold
            ):
                # add the article to the event and update the values
                event.add_article(article)
                assigned_to_event = True
                # TODO: merge and split the events
                break

            if len(sims) == idx + 1:
                # there are no more events to check
                break

            # go to the next closest event
            idx += 1

        if not assigned_to_event:
            # create a new news event cluster
            event = NewsEvent(articles=[article], use_ne=self.compare_named_entities)
            self.active_events.append(event)

        # remove events that are old
        self.__update_past_events(article.time)

    @property
    def events(self):
        """Get all of the events"""
        return self.past_events + self.active_events

    # ==================================
    # Statistics Methods
    # ==================================

    def event_centroid_distance(self):
        """Calculates the event similarities"""
        C = torch.cat(tuple([event.centroid.unsqueeze(0) for event in self.events]), 0)
        return torch.matmul(C, C.T).numpy()

    # ==================================
    # Remove Methods
    # ==================================

    # TODO: implement remove methods
    def __update_past_events(self, time):
        """Update the past events"""
        for event_id in reversed(range(len(self.active_events))):
            event = self.active_events[event_id]
            if self.time_threshold <= self.__absolute_difference(
                time, event.time_interval[self.time_compare]
            ):
                # add the event to the past events
                self.past_events.append(event)
                # remove the event from the active events
                del self.active_events[event_id]

    # ==================================
    # Merge Methods
    # ==================================

    # TODO: implement merge methods

    # ==================================
    # Split Methods
    # ==================================

    # TODO: implement split methods

    # ==================================
    # Evaluation Methods
    # ==================================

    def assign_events_to_articles(self):
        """Assigns the articles associated event ID"""
        # iterate through every articles
        events = self.past_events + self.active_events
        langs = set([event.lang for event in events])

        # iterate through each possible language
        for lang in langs:
            # get the language events
            lang_events = [event for event in events if event.lang == lang]
            for idx, event in enumerate(lang_events):
                for article in event.articles:
                    # the cluster ID is based on the language and the index
                    article.cluster_id = f"{lang}-{idx}"

    # ==================================
    # Helper Methods
    # ==================================

    def __has_similar_entities(
        self, e_entities: set, a_entities: set, threshold: int = 0.2
    ) -> bool:
        """Validates if the event and article have similar entities"""
        if len(e_entities) == 0 or len(a_entities) == 0:
            # cannot validate; use only content and time for validation
            return True

        # # measure the entities overlap
        j_index = jaccard_index(e_entities, a_entities)
        return j_index >= threshold
        # return len(e_entities & a_entities) >= threshold

    def __absolute_difference(self, time1, time2):
        """ "Calculates the absolute difference between two times"""
        return abs(time1 - time2)
