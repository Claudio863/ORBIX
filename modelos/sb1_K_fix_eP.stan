// Revision 2026-10-27: fase en un periodo y referencia temporal ponderada por precision.
// sb1_K_fix_eP.stan - ORBIX
// Modelo SB1 (una componente, K), esquema: P y e fijos.
// Priors configurables via bloque data; solo limites fisicos.
// PARAMETRIZACION DE MUESTREO (no cambia el modelo ni los priors):
//   lam_ref : longitud media w + M en t_ref (epoca media de los datos), real periodico.
//             Queda bien determinada aunque e -> 0 y casi no se correlaciona con P.
//   hk      : (h, k) = r (cos w, sin w), con e = 0.999 r^2 / (1 + r^2), w = atan2(k, h).
//             Coordenadas cartesianas: sin singularidad en e = 0 ni bordes en w = 0/2pi.
//             El termino -2 log(1 + r^2) es el jacobiano que hace e ~ uniforme(0, 0.999) y
//             w ~ uniforme(0, 2pi); el prior beta sobre e se suma encima.
//   T0, w, e son cantidades derivadas y cumplen exactamente sus limites fisicos:
//   T0 en [0, 1), w en [0, 2pi), 0 <= e < 0.999.
functions {
    // Ecuacion de Kepler resuelta en C++ (kepler_eq.hpp, gradientes analiticos)
    real kepler_eq(real M, real e);

    // Velocidad radial kepleriana. sgn = +1 primaria, -1 secundaria.
    // t: epocas relativas a t_ref; Mref: anomalia media en t_ref.
    // La amplitud K entra directa: es lo unico que las RV pueden medir.
    vector radial_velocity(int N, vector t, real Mref, real P, real e, real w,
                           real V0, real K, real sgn) {
        vector[N] E;
        vector[N] nu;
        for (j in 1:N) {
            E[j] = kepler_eq(2 * pi() * t[j] / P + Mref, e);    // anomalia excentrica
        }
        nu = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(E / 2));    // anomalia verdadera
        return V0 + sgn * K * (cos(w + nu) + e * cos(w));
    }
}
data {
    // Observaciones. Epocas en la UNIDAD DE TRABAJO elegida en la corrida
    // (dias o anios); P y T quedan en esa misma unidad.
    int<lower=1>   N_v1;
    vector[N_v1]   t_v1;
    vector[N_v1]   v1_obs;              // km/s
    vector<lower=0>[N_v1] v1_err;       // km/s
    real<lower=0>  P_fixed;             // periodo fijo (unidad de trabajo)
    real<lower=0, upper=0.999> e_fixed; // excentricidad fija

    // Cotas amplias de muestreo para priors uniformes propios.
    real V0_min;
    real<lower=V0_min> V0_max;
    real<lower=0> K_max;
    // Hiperparametros de los priors (vienen de config/priors.csv).
    // *_fam: 0 = sin prior explicito (plano dentro del limite fisico),
    //        1 = familia por defecto, 2 = alternativa (solo amplitudes).
    int<lower=0, upper=1> prior_T0_fam;     // 1: T0 ~ beta(a_T0, b_T0)
    real<lower=0> a_T0;
    real<lower=0> b_T0;
    int<lower=0, upper=1> prior_V0_fam;     // 1: V0 ~ normal(mu_V0, sigma_V0)
    real mu_V0;
    real<lower=0> sigma_V0;
    int<lower=0, upper=2> prior_K_fam;     // 1: K ~ normal(mu, sigma) T[0,] ; 2: K ~ lognormal(mu, sigma)
    real mu_K;
    real<lower=0> sigma_K;
}
transformed data {
    real min_t = min(t_v1);                    // primera epoca (origen de T0)
    real t_ref = dot_product(t_v1, inv_square(v1_err)) / sum(inv_square(v1_err));                   // epoca media (origen de lam_ref)
    vector[N_v1] tt1 = t_v1 - t_ref;
}
parameters {
    // Solo limites FISICOS (sin cotas artificiales de busqueda); ver nota arriba.
    real<lower=0, upper=2*pi()> lam_ref;  // longitud media en t_ref (rad, periodica)
    real                        w_raw;    // argumento del periastro (rad, periodico)
    real<lower=V0_min, upper=V0_max> V0;       // velocidad del centro de masa (km/s)
    real<lower=0, upper=K_max> K;       // semiamplitud (km/s)
}
transformed parameters {
    real P = P_fixed;                                     // fijo
    real e = e_fixed;                                  // fijo
    real w = w_raw;
    w = w - 2 * pi() * floor(w / (2 * pi()));               // w en [0, 2pi)
    real Mref = lam_ref - w;                                // anomalia media en t_ref
    // fase del periastro respecto de la primera epoca, en [0, 1)
    real T0 = (t_ref - min_t) / P - Mref / (2 * pi());
    T0 = T0 - floor(T0);
}
model {
    // --- priors ---
    // T0 es derivado; dado w es lineal en lam_ref (jacobiano constante), asi que
    // sumar su log-densidad equivale a poner el prior sobre T0. lam_ref: uniforme.
    if (prior_T0_fam == 1) target += beta_lpdf(T0 | a_T0, b_T0);
    if (prior_V0_fam == 1) V0 ~ normal(mu_V0, sigma_V0);
    // amplitudes: normal truncada en 0 (el limite <lower=0> trunca; la constante
    // de normalizacion es fija porque los hiperparametros son datos) o lognormal
    if (prior_K_fam == 1) K ~ normal(mu_K, sigma_K);
    else if (prior_K_fam == 2) K ~ lognormal(mu_K, sigma_K);

    // --- verosimilitud ---
    v1_obs ~ normal(radial_velocity(N_v1, tt1, Mref, P, e, w, V0, K, 1), v1_err);
}
generated quantities {
    real w_deg = w * 180 / pi();      // grados
    real T     = T0 * P + min_t;      // epoca del periastro (unidad de trabajo)
}


