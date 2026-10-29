#include <cmath>
#include <ostream>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun.hpp>   // para domain_error
#include <stan/math/prim/meta.hpp>  // para return_type_t

namespace sb2_fix_eP_model_namespace {

	using namespace stan::math;

	// Nueva función kepler genérica con comprobación de dominio
	template <typename T1, typename T2>
	stan::return_type_t<T1,T2>
	kepler(const T1& M, const T2& e, std::ostream* pstream) {

		using T = stan::return_type_t<T1,T2>;

		if (e < T2(0) || e >= T2(1))
			stan::math::domain_error("kepler","eccentricity is",e,"","");

		T E0 = M, E = M;
		for (int i = 0; i < 200; ++i) {
			T g  = E0 - e*std::sin(E0) - M;
			T gp = T(1) - e*std::cos(E0);
			E = E0 - g/gp;
			if (fabs((E - E0)/E) <= 1.234e-10) return E;
			E0 = E;
		}
		return E;          // mejor estimación si no converge
	}

	inline double kepler_eq(double M, double e, std::ostream* pstream) {
		double En;
		double E = M;
		for (int n = 0; n < 200; ++n) {
			// Newton's method iteration
			En = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
			// Convergence check
			if (fabs((En - E) / En) <= 1.234e-10) {
				return En;
			}
			E = En;
		}
		// Return the best estimate despite non-convergence
		return En;
	}

	inline var kepler_eq(const var& M_var, const var& e_var, std::ostream* pstream) {
		// Extract values from inputs
	  	double M = M_var.val();
		double e = e_var.val();
		double E = kepler_eq(M, e, pstream);
		// Compute the partial derivatives
		double dE_dM = 1.0 / (1.0 - e * cos(E));
		double dE_de = sin(E) * dE_dM;
		// Return the new parameter with values of the function and gradients
		return make_callback_var(E, [M_var, e_var, dE_dM, dE_de](auto& vi) {
			M_var.adj() += vi.adj() * dE_dM;
			e_var.adj() += vi.adj() * dE_de;
		});
	}

	inline var kepler_eq(double M, const var& e_var, std::ostream* pstream) {
		// Extract values from inputs
	  	double e = e_var.val();
		double E = kepler_eq(M, e, pstream);
		// Compute the partial derivatives
		double dE_de = sin(E) / (1.0 - e * cos(E));
		// Return the new parameter with values of the function and gradients
		return make_callback_var(E, [e_var, dE_de](auto& vi) {
			e_var.adj() += vi.adj() * dE_de;
		});
	}

	inline var kepler_eq(const var& M_var, double e, std::ostream* pstream) {
		// Extract values from inputs
	  	double M = M_var.val();
		double E = kepler_eq(M, e, pstream);
		// Compute the partial derivatives
		double dE_dM = 1.0 / (1.0 - e * cos(E));
		// Return the new parameter with values of the function and gradients
		return make_callback_var(E, [M_var, dE_dM](auto& vi) {
			M_var.adj() += vi.adj() * dE_dM;
		});
	}
}
