functions {
    real kepler_eq(real M, real e);
    vector radial_velocity(int N, vector t, real T, real P, real e, real w, real V0, real K) {
        // Variables declaration
        real M; vector[N] E; vector[N] nu; vector[N] V;
//        real convCoeff = 149597870.660 / (365.25 * 86400.0); // [AU] to [arcsec] 
        // Iterate over epochs
        for (j in 1:N) {
            // Mean anomaly
            M = 2 * pi() * (t[j] - T) / P;
            // Eccentric anomaly
            E[j] = kepler_eq(M, e);
        }
        // True anomaly
        nu = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(E / 2));
        // Primary object radial velocity
        // Radial velocity
        V = V0 + K * (cos(w + nu) + e * cos(w));
        return V;
    }
}
data {
    int<lower=0>   N_v1;       // number of samples
    vector[N_v1]   t_v1;       // epochs (years)
    vector[N_v1]   v1_obs;     // rv1 observations
    vector[N_v1]   v1_err;     // rv1 observation uncertainties
    real<lower=0,upper=1> e_fixed;   // fixed eccentricity (unificado: e_fixed)
    
}
transformed data {
    real       min_t = min(t_v1);    // first epoch
    real       min_RV = min(v1_obs);
    real       max_RV = max(v1_obs);
    real       delta_RV = max_RV-min_RV;
    vector[N_v1] t10 = t_v1 - min_t;
                      // normalized RV1 epochs   
}
parameters {
    real<lower=0, upper=1>             T0;      // normalized time of periastron passage (T - t0) / P
    real                               log_P;   // ln(period)
    real<lower=0, upper=2*pi()>        w;       // argument of periapsis (rad)
    real<lower=min_RV, upper=max_RV>   V0;      // velocity of center of mass (km/s)
    real<lower=0, upper=delta_RV>      K;      // amplitude (km/s)    
}
transformed parameters {
    real P = exp(log_P);   // period
}
model {
    // Variables declaration
    real T; 
    vector[N_v1] V1; 
    // Period
    T = T0 * P;
    // Primary object radial velocity
    V1 = radial_velocity(N_v1, t10, T, P, e_fixed, w, V0, K);
    v1_obs ~ normal(V1, v1_err);
}
generated quantities {
    real w_deg     = w * 180 / pi();                      // argument of periapsis (degrees)
    real T         = T0 * P + min_t;                      // unnormalized time of periastron passage
}
