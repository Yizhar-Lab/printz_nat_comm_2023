# %%
# from functions import make_m_rate_c
import logging
import pymc3 as pm
from pathlib import Path
import pandas as pd
import arviz as az
import xarray as xr

# %%


class MixtureModelInference():
    def __init__(self, data, priors, sample_params):
        self.set_priors(priors)

        rename_dict={a:i for i,a in enumerate(data["inter_rate"].rep_num.unique())}
        self.df_inter_rate = data["inter_rate"]
        self.df_inter_rate.rep_num=self.df_inter_rate.rep_num.apply(lambda a :rename_dict[a] )
        #self.df_inter_rate = data["inter_rate"]
        
        
        
        self.df_event_time = data["event_time"]
        self.df_event_time.rep_num=self.df_event_time.rep_num.apply(lambda a :rename_dict[a] )
        
        self.sample_params = sample_params
        self.models = {}
        self.traces = {}

        self.coords = {"reps": self.df_inter_rate.rep_num.unique()}
        self.modelfuncs = {
            "m_rate_nc": self.make_m_rate_nc,
            "m_rate_c": self.make_m_rate_c,

            "m_times_nc": self.make_m_times_nc,  # removed because has no free paramteres
            "m_times_c": self.make_m_times_c,

            "m_both_nc": self.make_m_both_nc,
            "m_both_c": self.make_m_both_c,
            "m_both_c_variable": self.make_m_both_c_variable,
        }

    def set_priors(self, priors):
        self.evoked_window = priors["evoked_window"]
        self.rate_mu = priors["rate_mu"]
        self.rate_sigma = priors["rate_sigma"]
        self.evoked_per_trial_mu = priors["evoked_per_trial_mu"]
        self.evoked_per_trial_sigma = priors["evoked_per_trial_sigma"]
        self.bump_center_mu = priors["bump_center_mu"]

        self.bump_width_mu = priors["bump_width_mu"]
        self.bump_center_sigma = priors["bump_center_sigma"]
        self.bump_width_sigma = priors["bump_width_sigma"]
        self.beta_alpha = priors["beta_alpha"]
        self.beta_beta = priors["beta_beta"]

    def fit(self):
        for model_name in self.modelfuncs:
            logging.info(f"\n=========\ncompiling {model_name}\n=========\n")
            self.models[model_name] = self.modelfuncs[model_name]()

            logging.info(f"\n=========\nsampling {model_name}\n=========\n")
            self.traces[model_name] = self.sample(model_name)

            logging.info(f"\n=========\nfinished sampling {model_name}\n=========\n")

    def sample(self, model_name):
        with self.models[model_name]:
            trace_model = pm.sample(**self.sample_params)
        return trace_model

    def make_all_dfs(self):
        logging.info(f"\n=========\n making dfs \n=========\n")
        self.make_summary_df(kind="all")  # just save everthing what is the downside?
        self.make_loo_waic_df()
        self.make_compare_df()
        logging.info(f"\n=========\n finished making dfs \n=========\n")

    def make_summary_df(self, kind="stats"):
        summaries = {}
        for name, trace in self.traces.items():
            summaries[name] = az.summary(trace, round_to="none", kind=kind)
            # summaries[name] = az.summary(trace, round_to="none", kind="stats").T.stack()
        self.summary_df = pd.concat(summaries).stack()
        # return summaries_df

    def make_loo_waic_df(self):
        ic_dict = {}
        for model_name in self.models:
            trace, model = self.traces[model_name], self.models[model_name]
            obs_vars = [str(o).split(' ')[0] for o in model.observed_RVs]
            for ov in obs_vars:
                # ic_dict[f"{model_name}:loo:{ov}"]= az.loo(trace,var_name=ov)
                ic_dict[(model_name, "loo", ov)] = az.loo(trace, var_name=ov)
                # ic_dict[f"{model_name}:waic:{ov}"]= az.waic(trace,var_name=ov)
                ic_dict[(model_name, "waic", ov)] = az.waic(trace, var_name=ov)
        self.loo_waic_df = pd.DataFrame.from_dict(ic_dict).stack([0, 1, 2]).reorder_levels([2, 0, 1, 3]).sort_index()
        # return ic_dict

    def make_compare_df(self, method="BB-pseudo-BMA", ic="loo"):
        """[summary]

        Args:
            method (str, optional): could be "stacking" too. Defaults to "BB-pseudo-BMA".
            ic (str, optional): could be "waic" too. Defaults to "loo".
        """
        # make rate comparision
        rate_compare_dict = {"connected": self.traces["m_rate_c"],
                             "not_connected": self.traces["m_rate_nc"]}
        rate_compare_df = az.compare(
            rate_compare_dict,
            method=method,
            ic=ic
        )
        # make times comparision
        times_compare_dict = {"connected": self.traces["m_times_c"],
                              "not_connected": self.traces["m_times_nc"]}
        times_compare_df = az.compare(
            times_compare_dict,
            method=method,
            ic=ic
        )

        # make both comparision
        self.make_total_ll("m_both_nc")
        self.make_total_ll("m_both_c")
        self.make_total_ll("m_both_c_variable")

        both_compare_dict = {"connected": self.traces["m_both_c"],
                             "not_connected": self.traces["m_both_nc"]}
        both_compare_df = az.compare(
            both_compare_dict,
            method=method,
            ic=ic
        )

        # make both comparision with variable
        variable_compare_dict = {"connected": self.traces["m_both_c"],
                                 "connected_variable": self.traces["m_both_c_variable"],
                                 "not_connected": self.traces["m_both_nc"]}
        variable_compare_df = az.compare(
            variable_compare_dict,
            method=method,
            ic=ic
        )

        self.drop_total_ll("m_both_c_variable")
        self.drop_total_ll("m_both_nc")
        self.drop_total_ll("m_both_c")

        # combine dfs
        compare_dict = {
            "rate": rate_compare_df,
            "times": times_compare_df,
            "both": both_compare_df,
            "both_var": variable_compare_df
        }
        self.compare_df = pd.concat(compare_dict).stack().reorder_levels([0, 2, 1]).sort_index()

    def make_total_ll(self, model_name):
        trace = self.traces[model_name]
        ll = trace.log_likelihood
        temp_rate = ll.rate_likelihood.rename({"rate_likelihood_dim_0": "logl"})
        temp_mixture = ll.mixture_likelihood.rename({"mixture_likelihood_dim_0": "logl"})

        trace.sample_stats["log_likelihood"] = xr.concat((temp_rate, temp_mixture), dim="logl")

    def drop_total_ll(self, model_name):
        trace = self.traces[model_name]
        trace.sample_stats = trace.sample_stats.drop("log_likelihood")

    def make_m_rate_nc(self):
        with pm.Model(coords=self.coords) as m_rate_nc:
            # data
            int_rep_nums = pm.Data('rep_nums', self.df_inter_rate.rep_num)
            tlen = pm.Data('tlen', self.df_inter_rate.tlen)
            num_events = pm.Data('num_events', self.df_inter_rate.num_events)

            # prior
            spont_rate = pm.Gamma(
                "spont_rate",
                mu=self.rate_mu,
                sigma=self.rate_sigma,
                dims="reps"
            )

            # likelihood
            rate_likelihood = pm.Poisson(
                'rate_likelihood',
                mu=spont_rate[int_rep_nums]*tlen,
                observed=num_events
            )

        # self.models["m_rate_nc"] = m_rate_nc
        return m_rate_nc

    def make_m_rate_c(self):
        with pm.Model(coords=self.coords) as m_rate_c:
            # data
            int_rep_nums = pm.Data('rep_nums', self.df_inter_rate.rep_num)
            tlen = pm.Data('tlen', self.df_inter_rate.tlen)
            num_events = pm.Data('num_events', self.df_inter_rate.num_events)
            is_evoked = pm.Data(
                'is_evoked',
                self.df_inter_rate.is_evoked.to_numpy().astype("int")
            )

            # prior
            evoked_per_trial = pm.Gamma(
                "evoked_per_trial",
                mu=self.evoked_per_trial_mu,
                sigma=self.evoked_per_trial_sigma
            )
            spont_rate = pm.Gamma(
                "spont_rate",
                mu=self.rate_mu,
                sigma=self.rate_sigma,
                dims="reps"
            )

            # likelihood
            rate_likelihood = pm.Poisson(
                'rate_likelihood',
                mu=spont_rate[int_rep_nums]*tlen + is_evoked * evoked_per_trial,
                observed=num_events
            )
        # self.models["m_rate_c"] = m_rate_c
        return m_rate_c

    def make_m_times_nc(self):
        with pm.Model(coords=self.coords) as m_times_nc:
            # data
            # event_rep_nums = pm.Data('event_rep_nums', self.df_event_time.rep_num)
            event_times = pm.Data('event_times', self.df_event_time.time_from_light)

            # dummy prior
            dummy = pm.Uniform("Dummy", 0, 1)
            # intermediate
            bump = pm.Gamma.dist(mu=self.bump_center_mu*1e-3, sigma=self.bump_width_mu*1e-3)
            unif = pm.Uniform.dist(lower=0, upper=self.evoked_window)

            # likelihood
            mixture_likelihood = pm.Mixture(
                'mixture_likelihood',
                w=[0.0, 1.0],
                comp_dists=[bump, unif], observed=event_times
            )
        # self.models["m_times_nc"] = m_times_nc
        return m_times_nc

    def make_m_times_c(self):
        with pm.Model(coords=self.coords) as m_times_c:
            # data
            event_rep_nums = pm.Data('event_rep_nums', self.df_event_time.rep_num)
            event_times = pm.Data('event_times', self.df_event_time.time_from_light)

            # priors
            w = pm.Beta('w', alpha=self.beta_alpha, beta=self.beta_beta, dims="reps")

            # intermediate
            bump = pm.Gamma.dist(mu=self.bump_center_mu*1e-3, sigma=self.bump_width_mu*1e-3)
            unif = pm.Uniform.dist(lower=0, upper=self.evoked_window)
            w2 = pm.math.stack(w, 1.0-w).T

            # likelihood
            mixture_likelihood = pm.Mixture(
                'mixture_likelihood',
                w=w2[event_rep_nums, :],
                comp_dists=[bump, unif],
                observed=event_times
            )
        # self.models["m_times_c"] = m_times_c
        return m_times_c

    def make_m_both_nc(self):
        with pm.Model(coords=self.coords) as m_both_nc:
            # data
            int_rep_nums = pm.Data('rep_nums', self.df_inter_rate.rep_num)
            tlen = pm.Data('tlen', self.df_inter_rate.tlen)
            num_events = pm.Data('num_events', self.df_inter_rate.num_events)
            # is_evoked = pm.Data(
            #     'is_evoked',
            #     self.df_inter_rate.is_evoked.to_numpy().astype("int")
            # )

            event_rep_nums = pm.Data('event_rep_nums', self.df_event_time.rep_num)
            event_times = pm.Data('event_times', self.df_event_time.time_from_light)

            # priors
            spont_rate = pm.Gamma(
                "spont_rate",
                mu=self.rate_mu,
                sigma=self.rate_sigma,
                dims="reps"
            )

            # intermediate
            bump = pm.Gamma.dist(mu=self.bump_center_mu*1e-3, sigma=self.bump_width_mu*1e-3)
            unif = pm.Uniform.dist(lower=0, upper=self.evoked_window)

            # likelihood
            rate_likelihood = pm.Poisson(
                'rate_likelihood',
                mu=spont_rate[int_rep_nums]*tlen,
                observed=num_events
            )
            mixture_likelihood = pm.Mixture(
                'mixture_likelihood',
                w=[0.0, 1.0],
                comp_dists=[bump, unif], observed=event_times
            )
        # self.models["m_both_nc"] = m_both_nc
        return m_both_nc

    def make_m_both_c(self):
        with pm.Model(coords=self.coords) as m_both_c:
            # data
            int_rep_nums = pm.Data('rep_nums', self.df_inter_rate.rep_num)
            tlen = pm.Data('tlen', self.df_inter_rate.tlen)
            num_events = pm.Data('num_events', self.df_inter_rate.num_events)
            is_evoked = pm.Data(
                'is_evoked',
                self.df_inter_rate.is_evoked.to_numpy().astype("int")
            )

            event_rep_nums = pm.Data('event_rep_nums', self.df_event_time.rep_num)
            event_times = pm.Data('event_times', self.df_event_time.time_from_light)

            # priors
            evoked_per_trial = pm.Gamma(
                "evoked_per_trial",
                mu=self.evoked_per_trial_mu,
                sigma=self.evoked_per_trial_sigma
            )

            spont_rate = pm.Gamma(
                "spont_rate",
                mu=self.rate_mu,
                sigma=self.rate_sigma,
                dims="reps"
            )

            # intermediate
            bump = pm.Gamma.dist(mu=self.bump_center_mu*1e-3, sigma=self.bump_width_mu*1e-3)
            unif = pm.Uniform.dist(lower=0, upper=self.evoked_window)

            w = pm.Deterministic("w",
                                 evoked_per_trial/(spont_rate*self.evoked_window + evoked_per_trial))
            w2 = pm.math.stack(w, 1.0-w).T

            # likelihood
            rate_likelihood = pm.Poisson(
                'rate_likelihood',
                mu=spont_rate[int_rep_nums]*tlen + is_evoked * evoked_per_trial,
                observed=num_events
            )

            # likelihood
            mixture_likelihood = pm.Mixture(
                'mixture_likelihood',
                w=w2[event_rep_nums, :],
                comp_dists=[bump, unif],
                observed=event_times
            )
        # self.models["m_both_c"] = m_both_c
        return m_both_c

    def make_m_both_c_variable(self):
        with pm.Model(coords=self.coords) as m_both_c_variable:
            # data
            int_rep_nums = pm.Data('rep_nums', self.df_inter_rate.rep_num)
            tlen = pm.Data('tlen', self.df_inter_rate.tlen)
            num_events = pm.Data('num_events', self.df_inter_rate.num_events)
            is_evoked = pm.Data(
                'is_evoked',
                self.df_inter_rate.is_evoked.to_numpy().astype("int")
            )

            event_rep_nums = pm.Data('event_rep_nums', self.df_event_time.rep_num)
            event_times = pm.Data('event_times', self.df_event_time.time_from_light)

            # priors
            bump_center = pm.Gamma(
                'bump_center',
                mu=self.bump_center_mu,
                sigma=self.bump_center_sigma)
            BoundedGamma = pm.Bound(
                pm.Gamma,
                lower=3.0,
                upper=8.0
            )
            bump_width = BoundedGamma(
                'bump_width',
                mu=self.bump_width_mu,
                sigma=self.bump_width_sigma
            )

            evoked_per_trial = pm.Gamma(
                "evoked_per_trial",
                mu=self.evoked_per_trial_mu,
                sigma=self.evoked_per_trial_sigma
            )
            spont_rate = pm.Gamma(
                "spont_rate",
                mu=self.rate_mu,
                sigma=self.rate_sigma,
                dims="reps"
            )

            # intermediate
            bump = pm.Normal.dist(mu=bump_center*1e-3, sigma=bump_width*1e-3)
            unif = pm.Uniform.dist(lower=0, upper=self.evoked_window)

            w = pm.Deterministic(
                "w",
                evoked_per_trial/(spont_rate*self.evoked_window + evoked_per_trial)
            )
            w2 = pm.math.stack(w, 1.0-w).T

            # likelihood
            rate_likelihood = pm.Poisson(
                'rate_likelihood',
                mu=spont_rate[int_rep_nums]*tlen + is_evoked * evoked_per_trial,
                observed=num_events
            )

            mixture_likelihood = pm.Mixture(
                'mixture_likelihood',
                w=w2[event_rep_nums, :],
                comp_dists=[bump, unif],
                observed=event_times
            )
        # self.models["m_both_c_variable"] = m_both_c_variable
        return m_both_c_variable


# %%


# %%
