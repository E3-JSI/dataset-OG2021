import torch
import itertools

from src.utils.NewsEvent import NewsEvent
from src.utils.NewsArticle import NewsArticle
from src.utils.LinearAlgebra import (
    cosine_similarity,
    avg_cosine_similarity,
    jaccard_index,
)

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

    def __init__(
        self,
        sim_threshold: float = 0.5,
        time_threshold_in_days: int = 1,
        time_compare_stat: str = "avg",
        compare_ne: bool = True,
    ) -> None:
        self.active_events = []
        self.past_events = []
        self.sim_threshold = sim_threshold
        self.time_threshold = time_threshold_in_days * ONE_DAY
        self.time_compare = time_compare_stat
        self.compare_named_entities = compare_ne

    # ==================================
    # Default Override Methods
    # ==================================

    # ==================================
    # Class Methods
    # ==================================

    def update(self, article: NewsArticle):
        """Update the events with the new article"""

        if len(self.active_events) == 0:
            # create a new news event cluster
            event = NewsEvent(articles=[article])
            self.active_events.append(event)
            return

        # TODO: create a time filter first for the events
        # TODO: older events put into a separate array (will not be used anymore)
        # TODO: use only events that are newer

        # calculate the similarity of the article to the events
        sims = torch.Tensor(
            [
                cosine_similarity(event.centroid, article.get_content_embedding())
                for event in self.active_events
            ]
        )
        # get the sorted indices of the most similar events
        sort_index = torch.argsort(sims, descending=True)

        idx = 0
        assigned_to_event = False
        while sims[sort_index[idx]] > self.sim_threshold:
            # get the next closest news event
            event = self.active_events[sort_index[idx]]

            # precalculate if the event and articles have similar entities
            has_similar_entities = (
                self.__has_similar_entities(
                    event.named_entities, article.get_named_entities()
                )
                if self.compare_named_entities
                else True
            )

            if has_similar_entities and self.time_threshold <= 0:
                event.add_article(article)
                assigned_to_event = True
                # TODO: merge and split the events
                break

            # check if the article happened at an approximate
            # same time as the rest of the event articles
            time_diff = self.__absolute_difference(
                article.time, event.time_interval[self.time_compare]
            )
            if has_similar_entities and time_diff <= self.time_threshold:
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
            event = NewsEvent(articles=[article])
            self.active_events.append(event)

        # remove events that are old
        self.__update_past_events(article.time)

    def get_events(self):
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
        for idx, event in enumerate(events):
            for article in event.articles:
                article.cluster_id = idx

    def measure_performance(self):
        """Measures the performance of the clustering algorithm"""
        # get all of the articles in one array
        events = self.past_events + self.active_events
        articles = list(itertools.chain(*[event.articles for event in events]))

        # get the following statistics
        # tp - number of correctly clustered-together article pairs
        # fp - number of incorrectly clustered-together article pairs
        # fn - number of incorrectly not-clustered-together article pairs
        # tn - number of correctly not-clustered-together article pairs
        tp, fp, fn, tn = 0, 0, 0, 0
        for i, ai in enumerate(articles):
            for aj in articles[i + 1 :]:
                if ai.event_id == aj.event_id and ai.cluster_id == aj.cluster_id:
                    tp += 1
                elif ai.event_id != aj.event_id and ai.cluster_id == aj.cluster_id:
                    fp += 1
                elif ai.event_id == aj.event_id and ai.cluster_id != aj.cluster_id:
                    fn += 1
                else:
                    tn += 1

        # get the precision, recall and F1 scores
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F1 = 2 * (P * R) / (P + R)

        # return the metrics
        return {"F1": F1, "P": P, "R": R}

    # ==================================
    # Helper Methods
    # ==================================

    def __has_similar_entities(
        self, e_entities: set, a_entities: set, threshold: int = 1
    ) -> bool:
        """Validates if the event and article have similar entities"""
        # if len(e_entities) == 0 or len(a_entities) == 0:
        #     # cannot validate; use only content and time for validation
        #     return True

        # # measure the entities overlap
        # j_index = jaccard_index(e_entities, a_entities)
        # return j_index >= threshold
        return len(e_entities & a_entities) >= threshold

    def __absolute_difference(self, time1, time2):
        """ "Calculates the absolute difference between two times"""
        return abs(time1 - time2)
