import torch
from src.utils.NewsEvent import NewsEvent
from src.utils.Wasserstein import Wasserstein
from src.utils.LinearAlgebra import cosine_similarity
from typing import List

# ===============================================
# Define constants
# ===============================================

# one day in seconds
ONE_DAY = 86400

# ===============================================
# Define the News Event Monitor
# ===============================================


class MultiNewsEventMonitor:
    active_events: List[NewsEvent]
    past_events: List[NewsEvent]
    sim_th: float
    time_th: float
    time_compare: str
    filter_cls: bool
    filter_cls_n: int
    device: torch.device

    def __init__(
        self,
        sim_th: float,
        time_th_in_days: float,
        w_reg: float = 0.1,
        w_nit: int = 100,
        filter_cls: bool = False,
        filter_cls_n: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.active_events = []
        self.past_events = []
        self.sim_th = sim_th
        self.time_th = time_th_in_days * ONE_DAY
        self.wasserstein = Wasserstein(reg=w_reg, nit=w_nit, device=device)
        self.filter_cls = filter_cls
        self.filter_cls_n = filter_cls_n

    # ==================================
    # Default Override Methods
    # ==================================

    # ==================================
    # Class Methods
    # ==================================

    def update(self, mono_event: NewsEvent):
        if len(self.active_events) == 0:
            # create a new news event cluster
            self.active_events.append(mono_event)
            return

        viewed_active_events = self.active_events
        if self.filter_cls:
            # filter based on most similar centroids
            tmp_s = torch.Tensor(
                [
                    cosine_similarity(event.centroid, mono_event.centroid)
                    for event in self.active_events
                ]
            )
            # update the viewed active events
            sort_index = torch.argsort(tmp_s, descending=True)
            viewed_active_events = [
                viewed_active_events[idx] for idx in sort_index[: self.filter_cls_n]
            ]

        # calculate the similarity of the monolingual events
        sims = []
        for multi_event in viewed_active_events:
            # calculate using wasserstein distance
            mo_emb = mono_event.get_article_embeddings(type="content")
            me_emb = multi_event.get_article_embeddings(type="content")
            C = self.wasserstein.get_cost_matrix(me_emb, mo_emb)
            me_dist = self.wasserstein.get_distributions(torch.ones(me_emb.shape[:2]))
            mo_dist = self.wasserstein.get_distributions(torch.ones(mo_emb.shape[:2]))
            sim, _, _ = self.wasserstein(C, me_dist, mo_dist, as_prob=True)
            sims.append(sim)

        sims = torch.Tensor(sims)
        # get the sorted indices of the most similar events
        sort_index = torch.argsort(sims, descending=True)

        idx = 0
        assigned_to_event = False
        while sims[sort_index[idx]] > self.sim_th:
            # get the next closest news event
            multi_event = viewed_active_events[sort_index[idx]]

            # check if the article happened at an approximate
            # same time as the rest of the event articles
            time_diff = self.__absolute_difference(
                multi_event.min_time, mono_event.min_time
            )
            if time_diff <= self.time_th:
                # add the article to the event and update the values
                multi_event.add_articles(mono_event.articles)
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
            self.active_events.append(mono_event)

        # remove events that are old
        self.__update_past_events(mono_event.min_time)

    @property
    def events(self):
        """Get all of the events"""
        return self.past_events + self.active_events

    # ==================================
    # Remove Methods
    # ==================================

    # TODO: implement remove methods

    def __update_past_events(self, time):
        """Update the past events"""
        for event_id in reversed(range(len(self.active_events))):
            event = self.active_events[event_id]
            if self.time_th <= self.__absolute_difference(time, event.min_time):
                # add the event to the past events
                self.past_events.append(event)
                # remove the event from the active events
                del self.active_events[event_id]

    # ==================================
    # Evaluation Methods
    # ==================================

    def merge_multi_event_clusters(self):
        """Merges the multi-events into a single event"""
        # iterate through every articles
        multi_events = self.past_events + self.active_events

        for idx, event in enumerate(multi_events):
            for article in event.articles:
                article.cluster_id = f"wn-{idx+1}"

        return multi_events

    def __absolute_difference(self, time1, time2):
        """ "Calculates the absolute difference between two times"""
        return abs(time1 - time2)
