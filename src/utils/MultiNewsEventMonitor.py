import torch
from src.utils.NewsEventBase import NewsEventBase
from src.utils.Wasserstein import Wasserstein
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
    active_events: List[NewsEventBase]
    past_events: List[NewsEventBase]
    sim_threshold: float
    time_threshold: float
    time_compare: str

    def __init__(
        self,
        sim_threshold: float,
        time_threshold_in_days: float,
        w_reg: float = 0.1,
        w_nit: int = 100,
    ) -> None:
        self.active_events = []
        self.past_events = []
        self.sim_threshold = sim_threshold
        self.time_threshold = time_threshold_in_days * ONE_DAY
        self.wasserstein = Wasserstein(reg=w_reg, nit=w_nit)

    # ==================================
    # Default Override Methods
    # ==================================

    # ==================================
    # Class Methods
    # ==================================

    def update(self, mono_event: NewsEventBase):
        if len(self.active_events) == 0:
            # create a new news event cluster
            self.active_events.append(mono_event)
            return

        # calculate the similarity of the monolingual events
        sims = []
        for multi_event in self.active_events:
            # calculate using wasserstein distance
            mo_emb = mono_event.get_article_embeddings()
            me_emb = multi_event.get_article_embeddings()
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
        while sims[sort_index[idx]] > self.sim_threshold:
            # get the next closest news event
            multi_event = self.active_events[sort_index[idx]]

            # check if the article happened at an approximate
            # same time as the rest of the event articles
            time_diff = self.__absolute_difference(
                multi_event.min_time, mono_event.min_time
            )
            if time_diff <= self.time_threshold:
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
            if self.time_threshold <= self.__absolute_difference(time, event.min_time):
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
