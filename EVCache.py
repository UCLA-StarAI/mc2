from scipy.special import comb
import numpy as np

class EVCache:

    def __init__(self):
        self.f_cache = dict()
        self.g_cache = dict()
        self.fg_cache = dict()

        self.exp_order = []

        self.moment_g_cache = dict()
        self.moment_fg_cache = dict()


    def get_f(self, psdd_id, lgc_id):
        return self._get("f", self.f_cache, psdd_id, lgc_id)
    def put_f(self, psdd_id, lgc_id, value):
        self._put(self.f_cache, psdd_id, lgc_id, value)

    def get_g(self, psdd_id, lgc_id):
        return self._get("g", self.g_cache, psdd_id, lgc_id)
    def put_g(self, psdd_id, lgc_id, value):
        self._put(self.g_cache, psdd_id, lgc_id, value)

    def get_fg(self, psdd_id, lgc_id):
        return self._get("fg", self.fg_cache, psdd_id, lgc_id)
    def put_fg(self, psdd_id, lgc_id, value):
        self._put(self.fg_cache, psdd_id, lgc_id, value)

    def get_moment_g(self, psdd_id, lgc_id, moment):
        if (moment, psdd_id, lgc_id) in self.moment_g_cache:
            return self.moment_g_cache[(moment, psdd_id, lgc_id)]
        return None
    def put_moment_g(self, psdd_id, lgc_id, moment, value):
        self.moment_g_cache[(moment, psdd_id, lgc_id)] = value

    def get_moment_fg(self, psdd_id, lgc_id, moment):
        if (moment, psdd_id, lgc_id) in self.moment_fg_cache:
            return self.moment_fg_cache[(moment, psdd_id, lgc_id)]
        return None
    def put_moment_fg(self, psdd_id, lgc_id, moment, value):
        self.moment_fg_cache[(moment, psdd_id, lgc_id)] = value

    ###### 
    def _get(self, types, cache, psdd_id, lgc_id):
        if (psdd_id, lgc_id) in cache:
            return cache[(psdd_id, lgc_id)]
        else:
            self.exp_order.append( (types, psdd_id, lgc_id) )
            return None    

    def _put(self, cache, psdd_id, lgc_id, value):
        cache[ (psdd_id, lgc_id) ] = value
    ######

    def clear(self):
        self.f_cache.clear()
        self.g_cache.clear()
        self.fg_cache.clear()
        self.moment_g_cache.clear()
        self.moment_fg_cache.clear()