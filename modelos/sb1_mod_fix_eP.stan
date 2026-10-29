functions {
  real kepler_eq(real M, real e);

  vector radial_velocity(int N, vector t, real T, real P, real e,
                         real w, real V0, real K) {
    real M;  vector[N] E;  vector[N] nu;  vector[N] V;
    for (j in 1:N) {
      M     = 2 * pi() * (t[j] - T) / P;
      E[j]  = kepler_eq(M, e);
    }
    nu = 2 * atan( sqrt((1+e)/(1-e)) * tan(E/2) );
    V  = V0 + K * (cos(w+nu) + e*cos(w));
    return V;
  }
}

data {
  int<lower=0>   N_v1;
  vector[N_v1]   t_v1;
  vector[N_v1]   v1_obs;
  vector[N_v1]   v1_err;

  real<lower=0,upper=1> e_fixed;    // 
  real<lower=0>         P_fixed;    // 
}

transformed data {
  real       min_t    = min(t_v1);
  real       min_RV   = min(v1_obs);
  real       max_RV   = max(v1_obs);
  real       delta_RV = max_RV - min_RV;
  vector[N_v1] t10    = t_v1 - min_t;
}

parameters {
  real<lower=0,upper=1>      T0;
  real<lower=0,upper=2*pi()> w;
  real<lower=min_RV,upper=max_RV> V0;
  real<lower=0,upper=delta_RV>    K;
}

model {
  real T       = T0 * P_fixed;
  vector[N_v1] V1 = radial_velocity(N_v1, t10, T, P_fixed,
                                    e_fixed, w, V0, K);
  v1_obs ~ normal(V1, v1_err);
}

generated quantities {
  real w_deg = w * 180 / pi();
  real T = T0 * P_fixed + min_t;
}
