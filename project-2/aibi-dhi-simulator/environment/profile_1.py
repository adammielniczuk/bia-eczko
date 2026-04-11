from gymnasium import Env
import numpy as np
from collections import deque
import random
import math
from gymnasium import error, spaces, utils


# one episode is one day


class Profile1(Env):

    def __init__(
        self,
        behavior_threshold=25,
        has_family=True,
        good_time=1,
        habituation=False,
        time_preference_update_step=100000000,
    ):
        self.behavior_threshold = behavior_threshold
        self.has_family = has_family
        self.good_time = good_time  # 0 morning, 1 midday, 2 evening, 3 night
        self.habituation = habituation
        self.time_preference_update_step = time_preference_update_step
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete(
            [2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 2, 25, 25]
        )
        self.activity_p = 0
        self.activity_s = 0
        self.hour_steps = 0
        self.env_steps = 0
        self.max_notification_tolerated = 3
        self.confidence_threshold = 4
        self.week_days = deque(np.arange(1, 8), maxlen=7)
        self.hours = deque(np.arange(0, 24), maxlen=24)
        self.rr = []

        self.valence_list = random.choices([0, 1], weights=(0.9, 0.1), k=23)
        self.arousal_list = random.choices([0, 1, 2], weights=(0.4, 0.2, 0.4), k=23)
        self.activity_performed = [0]
        self.num_performed = []
        self.num_notified = []
        self._start_time_randomiser()
        self.time_of_the_day = self.hours[0]
        self.day_of_the_week = self.week_days[0]  # 1 Monday, 7 Sunday
        self.motion_activity_list = random.choices(
            ["stationary", "walking"], weights=(1.0, 0.0), k=24
        )  # in last 24 hours
        self.awake_list = random.choices(
            ["sleeping", "awake"], weights=(0.15, 0.85), k=24
        )  # insomnia
        self.last_activity_score = np.random.randint(0, 2)  # 0 negative, 1 positive
        self.location = (
            "home"
            if 1 < self.time_of_the_day < 7
            else np.random.choice(["home", "other"])
        )
        self.valence = 1
        self.arousal = 1
        self.cognitive_load = 0
        self._update_emotional_state()
        self._initialise_awake_probailities()
        self.h_slept = []
        self.h_positive = []
        self.h_nonstationary = []
        self.observation_list = [self._get_current_state()]
        self.reset()
        # valence =1 #negative/ positive
        # arousal = 1 # low, mid, high
        # cognitive_load = 0 # low, high

        # sleeping=0 # no/yes
        # number_of_hours_slept, more 1 , 0 less than7
        # last_activity_score =1 # low/ high percived benefit
        # time_since_last_activity = 1 # less 0 or 1 more than 24 hours

        # location =0 # home/ other
        # motion_activity = 0 #stationary, walking

        # day_time = 0 # morning 6am-10am, midday 10am-6pm, evening 6pm-10pm, night 10pm-6am
        # week_day  = 1 # week day, weekend

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.activity_p = 0
        self.activity_s = 0
        self.hour_steps = 0
        return self._get_current_state(), self._get_current_info(action=0)

    def update_after_day(self):
        if self.activity_s != 0:
            self.rr.append(self.activity_p / self.activity_s)
        else:  # unsucessful training
            self.rr.append(np.nan)
            # self.num_notified.append(np.nan)
        self.num_notified.append(self.activity_s)
        self.num_performed.append(self.activity_p)
        self.h_slept.append(self.awake_list[-24:].count("sleeping"))
        self.h_positive.append(sum(self.valence_list[-24:]))
        self.h_nonstationary.append(self.motion_activity_list[-24:].count("walking"))
        self.reset()
        if self.habituation:
            self.behavior_threshold = (
                self.behavior_threshold + 0.15
            )  # increase the threshold for performing action

    def step(self, action: int):
        info = self._get_current_info(action)

        if action == 1:
            self.activity_s = self.activity_s + 1
            behavior = self.fogg_behavior(
                info["motivation"], info["ability"], info["trigger"]
            )
            self.observation_list.append(self._get_current_state())
            if behavior:
                self.activity_p = self.activity_p + 1
                self.activity_performed.append(1)
                self._update_patients_activity_score()
                reward = 20
            else:
                self.activity_performed.append(0)
                if self.activity_s < self.max_notification_tolerated:
                    reward = -1
                else:
                    reward = -10
        else:
            reward = 0.0
        # info['reward'] = reward

        self.update_state()
        self.hour_steps = self.hour_steps + 1
        self.env_steps = self.env_steps + 1
        if self.hour_steps == 24:
            self.update_after_day()
            terminated = True
        else:
            terminated = False
        if self.env_steps > self.time_preference_update_step:
            self.good_time = 2  # update time preference to be in the evening

        truncated = False
        state = self._get_current_state()
        return state, reward, terminated, truncated, info

    # Work-around for Gymnasium
    def _get_current_info(self, action):
        info = dict()
        info["motivation"] = self.get_motivation()
        info["ability"] = self.get_ability()
        info["trigger"] = self.get_trigger()
        info["action"] = action
        return info

    def _get_current_state(self):
        # valence =1 #negative/ positive
        # arousal = 1 # low, mid, high
        # cognitive_load = 0 # low, high

        # sleeping=0 # no/yes
        # number_of_hours_slept in last 24h sufficient 7
        # last_activity_score =1 # low/ high percived benefit
        # time_since_last_activity = 1 # less 0 or 1 more than 24 hours

        # location =0 # home/ other
        # motion_activity = 0 #stationary, walking

        # day_time = 0 # morning 6am-10am, midday 10am-6pm, evening 6pm-10pm, night 10pm-6am
        # week_day  = 1 # week day, weekend

        location = 1 if self.location == "home" else 0
        sleeping = 1 if self.awake_list[-1] == "sleeping" else 0
        motion_activity = 1 if self.motion_activity_list[-1] == "walking" else 0
        week_day = self._get_week_day()
        day_time = self._get_time_day()
        t = self._time_since_last_activity()
        number_of_hours_slept = 1 if self.awake_list[-24:].count("sleeping") >= 7 else 0

        obs = np.array(
            [
                self.valence,
                self.arousal,
                self.cognitive_load,
                sleeping,
                number_of_hours_slept,
                self.last_activity_score,
                t,
                location,
                motion_activity,
                day_time,
                week_day,
                self.activity_s,
                self.activity_p,
            ]
        )
        return obs

    def _time_since_last_activity(self):
        if self.activity_p == 0:
            return 1  # more than 24 hours
        else:
            return 0

    def fogg_behavior(self, motivation: int, ability: int, trigger: bool) -> bool:
        """ "
        Function that decides if the behavior will be performed or not based on Fogg's Behavioral Model
        """
        behavior = motivation * ability * trigger
        return behavior > self.behavior_threshold

    # CHANGED: not motivated by own success or breaking records
    def get_motivation(self):
        number_of_hours_slept = self.awake_list[-24:].count("sleeping")
        sufficient_sleep = 1 if number_of_hours_slept > 7 else 0
        return self.valence + self.has_family + sufficient_sleep

    def get_ability(self):
        n = self.activity_p  # 0  if the activity was already performed twice
        if n == 0:
            not_tired_of_repeating_the_activity = 1
        elif n == 1:
            not_tired_of_repeating_the_activity = 0
        else:
            not_tired_of_repeating_the_activity = -2  # CHANGED: stress affects ability
        ready = self._time_since_last_activity()
        load = 1 if self.cognitive_load == 0 else -1  # CHANGED: stress affects ability
        confidence = (
            1 if sum(self.activity_performed) >= self.confidence_threshold else 0
        )

        return confidence + load + not_tired_of_repeating_the_activity + ready

    def get_trigger(self):
        prompt = (
            1 if self.awake_list[-1] != "sleeping" else 0
        )  # do not prompt when patient sleep
        good_time = 1 if self._get_time_day() == self.good_time else 0
        good_day = 1 if self._get_week_day() == 1 else 0
        good_location = 1 if self.location == "home" else 0
        good_motion = 1 if self.motion_activity_list[-1] == "stationary" else 0
        good_arousal = 1 if self.arousal == 1 else 0

        return (
            good_arousal + good_day + good_time + good_location + good_motion
        ) * prompt

    def update_state(self):
        self._update_time()
        self._update_awake()
        if self.awake_list[-1] == "awake":
            self._update_motion_activity()
            self._update_location()
            self._update_emotional_state()
        else:
            self.location = "home"
            self.motion_activity_list.append("stationary")
            self.arousal = 0
            self.cognitive_load = 0
            self.valence_list.append(self.valence)
            self.arousal_list.append(self.arousal)

    def _update_day(self):

        self.week_days.rotate(-1)
        self.day_of_the_week = self.week_days[0]

    def _get_week_day(self):
        if self.day_of_the_week < 6:
            return 0  # week day
        else:
            return 1  # weekend

    def _get_time_day(self):
        if 10 >= self.time_of_the_day >= 6:
            return 0  # morning
        elif 18 > self.time_of_the_day >= 10:
            return 1  # midday
        elif 22 > self.time_of_the_day >= 18:
            return 2  # evening
        else:
            return 3  # night

    def _update_time(self):

        self.hours.rotate(-1)
        self.time_of_the_day = self.hours[0]
        if self.time_of_the_day == 0:
            self._update_day()

    def _start_time_randomiser(self):
        for i in range(np.random.randint(0, len(self.week_days))):
            self.week_days.rotate(-1)
        for i in range(np.random.randint(0, len(self.hours))):
            self.hours.rotate(-1)

    def _update_emotional_state(self):
        # random
        self._update_patient_stress_level()  # updates arousal and valence
        self._update_patient_cognitive_load()  # 0 low, 1 high

    def _update_motion_activity(self):

        if self.activity_performed[-1] == 1:
            weights = (0, 1)
        else:
            threshold = 0.3  # equivalent to 6 h daily
            w_r = self.motion_activity_list.count("walking") / len(
                self.motion_activity_list
            )
            w = w_r if w_r < threshold else threshold
            st = 1 - w
            weights = (st, w)
        self.motion_activity_list.append(
            random.choices(["stationary", "walking"], weights=weights, k=1)[0]
        )

    def _update_awake(self):
        """ "
        Fairholme & Manber (2015) "Sleep, emotions, and emotion regulation: an overview"
        "negative valence and high arousal potentially have unique effects on sleep architecture,
        with high arousal being associated with reductions in slow-wave sleep and negative valence being associated
        with disruptions to REM sleep"
           - negative valence (-)
           - high arousal (-)

        Bisson et al (2019) "Walk to a better night of sleep: testing the relationship between physical
        activity and sleep"
        " on days that participants were more active than average,
        they reported better sleep quality and duration in both sexes. Results suggest
        that low-impact PA is positively related to sleep, more so in women than men"
             - walking (+)"""

        if self.activity_p > 0:
            awake_prb = self.health_sleep[self.time_of_the_day]  # healthy sleeping
        else:
            if self.arousal == 2 and self.valence == 0:
                awake_prb = self.insomnia[self.time_of_the_day]  # insomnia
            else:
                awake_prb = self.semihealthy_sleep[self.time_of_the_day]  # semi-healthy

        now_awake = random.choices(
            ["sleeping", "awake"], weights=(1 - awake_prb, awake_prb), k=1
        )
        self.awake_list.append(now_awake[0])

    def _update_location(self):
        if self.motion_activity_list[-1] == "walking":
            self.location = "other"
        else:
            self.location = random.choices(["home", "other"], weights=(0.8, 0.2), k=1)[
                0
            ]

    @staticmethod
    def _prob_awake(x):
        x = x + 14
        return -0.5 * math.sin((x + 2) / 3.5) + 0.5

    def _awake_pattern(self, x, z):
        x = x - 14
        x = abs(x)
        return np.where(x <= 6, 0.98, self._prob_awake(x) + z)

    def _initialise_awake_probailities(self):

        self.health_sleep = [self._awake_pattern(x, 0.15) for x in range(0, 24)]
        self.semihealthy_sleep = [self._awake_pattern(x, 0.35) for x in range(0, 24)]
        self.insomnia = [self._awake_pattern(x, 0.6) for x in range(0, 24)]

    def _update_patient_stress_level(self):
        """ "
        Stress = high arousal and negative valence Markova et al (2019) "arousal-valence emotion space"
        in contrast to
        Flow = high/mid arousal and positive valence//

        Peifer et al (2014) "The relation of flow-experience and physiological arousal under stress — Can u shape it?"
        "Physiological arousal during flow-experience between stress and relaxation"

        1) Yoon et al (2014) "Understanding notification stress of smartphone messenger app"
         - number of notification high (stress +), low (stress -)

        2) Zhai et al (2020) "Associations among physical activity and smartphone use with perceived stress and sleep
        quality of Chinese college students"
         - insufficient exercise (stress +), exercise in past day (stress -)
        """

        insufficient_exercise = (
            1 if self.motion_activity_list[-24:].count("walking") < 1 else 0
        )
        annoyed = 1 if self.activity_s > self.max_notification_tolerated else 0
        number_of_hours_slept = self.awake_list[-24:].count("sleeping")
        insufficient_sleep = 1 if number_of_hours_slept < 7 else 0
        neg_factors = insufficient_exercise + annoyed + insufficient_sleep

        if self.motion_activity_list[-1] == "walking":
            self.valence, self.arousal = 1, 1
        else:
            if neg_factors >= 2:
                self.valence = 0
                self.arousal = 2
            elif neg_factors == 1:
                self.valence = random.choices([0, 1], weights=(0.5, 0.5), k=1)[0]
                self.arousal = random.choices([0, 1, 2], weights=(0.3, 0.3, 0.4), k=1)[
                    0
                ]
            else:
                self.valence = 1
                self.arousal = random.choices([0, 1, 2], weights=(0.3, 0.4, 0.3), k=1)[
                    0
                ]
        self.valence_list.append(self.valence)
        self.arousal_list.append(self.arousal)

    def _update_patient_cognitive_load(self):
        """ "

        Okoshi et al (2015)  "Attelia: Reducing User’s Cognitive Load due to Interruptive Notifications on Smart Phones"
        Okoshi et al (2017) "Attention and Engagement-Awareness in the Wild: A Large-Scale Study with Adaptive Notifications"
        "notifications at detected breakpoint timing resulted in 46% lower cognitive load compared to randomly-timed
         notifications"
        """
        if self.activity_s > 0:
            self.cognitive_load = 1 if self.activity_p / self.activity_s < 0.5 else 0
        else:
            self.cognitive_load = np.random.randint(0, 2)

    def _update_patients_activity_score(self):
        """ "
        Williams et al (2012) "Does Affective Valence During and Immediately Following a 10-Min Walk Predict Concurrent
         and Future Physical Activity?"
         "During-behavior affect is predictive of concurrent and future physical activity behavior"

        """
        self.last_activity_score = self.valence
